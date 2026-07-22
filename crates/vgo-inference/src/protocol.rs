use std::{
    collections::{HashMap, HashSet},
    io::{Cursor, Read},
    mem::size_of,
};

use vgo_raster::{CHANNEL_COUNT, SemanticRaster};
use vgo_search::EvaluationError;

const REQUEST_MAGIC: [u8; 8] = *b"VGOIFR01";
const RESPONSE_MAGIC: [u8; 8] = *b"VGOOFR01";
const PROTOCOL_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq)]
pub struct InferenceInput {
    id: u64,
    raster: SemanticRaster,
}

impl InferenceInput {
    #[must_use]
    pub const fn new(id: u64, raster: SemanticRaster) -> Self {
        Self { id, raster }
    }

    #[must_use]
    pub const fn id(&self) -> u64 {
        self.id
    }

    #[must_use]
    pub const fn raster(&self) -> &SemanticRaster {
        &self.raster
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct InferenceOutput {
    id: u64,
    current_value: f64,
    policy: Vec<f32>,
}

impl InferenceOutput {
    pub fn new(id: u64, current_value: f64, policy: Vec<f32>) -> Result<Self, EvaluationError> {
        if !current_value.is_finite() || !(-1.0..=1.0).contains(&current_value) {
            return Err(EvaluationError::new("invalid inference value"));
        }
        if policy.iter().any(|logit| !logit.is_finite()) {
            return Err(EvaluationError::new("non-finite policy logit"));
        }
        Ok(Self {
            id,
            current_value,
            policy,
        })
    }

    #[must_use]
    pub const fn id(&self) -> u64 {
        self.id
    }

    #[must_use]
    pub const fn current_value(&self) -> f64 {
        self.current_value
    }

    #[must_use]
    pub fn policy(&self) -> &[f32] {
        &self.policy
    }

    #[must_use]
    pub fn into_parts(self) -> (u64, f64, Vec<f32>) {
        (self.id, self.current_value, self.policy)
    }
}

pub fn encode_request_frame(batch: &[InferenceInput]) -> Result<Vec<u8>, EvaluationError> {
    let first = batch
        .first()
        .ok_or_else(|| EvaluationError::new("inference batch must not be empty"))?;
    let config = first.raster.config();
    let identifiers = batch.iter().map(InferenceInput::id).collect::<HashSet<_>>();
    if identifiers.len() != batch.len() {
        return Err(EvaluationError::new("duplicate inference request ID"));
    }
    if batch.iter().any(|input| input.raster.config() != config) {
        return Err(EvaluationError::new(
            "mixed raster shapes in inference batch",
        ));
    }

    let item_bytes = 8 + CHANNEL_COUNT * config.pixels() * size_of::<f32>();
    let mut frame = Vec::with_capacity(28 + batch.len() * item_bytes);
    frame.extend_from_slice(&REQUEST_MAGIC);
    push_u32(&mut frame, PROTOCOL_VERSION);
    push_u32(&mut frame, batch.len() as u32);
    push_u32(&mut frame, CHANNEL_COUNT as u32);
    push_u32(&mut frame, config.height as u32);
    push_u32(&mut frame, config.width as u32);
    for input in batch {
        frame.extend_from_slice(&input.id.to_le_bytes());
        for value in input.raster.data() {
            frame.extend_from_slice(&value.to_le_bytes());
        }
    }
    Ok(frame)
}

pub fn read_response_frame(
    reader: &mut impl Read,
    batch: &[InferenceInput],
) -> Result<Vec<InferenceOutput>, EvaluationError> {
    let config = batch
        .first()
        .ok_or_else(|| EvaluationError::new("inference batch must not be empty"))?
        .raster
        .config();
    let mut magic = [0_u8; 8];
    reader.read_exact(&mut magic).map_err(io_error)?;
    if magic != RESPONSE_MAGIC {
        return Err(EvaluationError::new("invalid inference response magic"));
    }
    let version = read_u32(reader)?;
    let count = read_u32(reader)? as usize;
    let policy_size = read_u32(reader)? as usize;
    if version != PROTOCOL_VERSION || count != batch.len() {
        return Err(EvaluationError::new("inference response header mismatch"));
    }
    if policy_size != config.pixels() + 1 {
        return Err(EvaluationError::new("inference policy size mismatch"));
    }

    let item_bytes = 8 + size_of::<f32>() + policy_size * size_of::<f32>();
    let mut body = vec![0_u8; count * item_bytes];
    reader.read_exact(&mut body).map_err(io_error)?;
    decode_response_body(&body, batch, policy_size)
}

fn decode_response_body(
    body: &[u8],
    batch: &[InferenceInput],
    policy_size: usize,
) -> Result<Vec<InferenceOutput>, EvaluationError> {
    let expected_ids = batch.iter().map(InferenceInput::id).collect::<HashSet<_>>();
    let mut reader = Cursor::new(body);
    let mut outputs = HashMap::with_capacity(batch.len());
    for _ in batch {
        let id = read_u64(&mut reader)?;
        if !expected_ids.contains(&id) {
            return Err(EvaluationError::new("unexpected inference response ID"));
        }
        let current_value = f64::from(read_f32(&mut reader)?);
        if !current_value.is_finite() || !(-1.0..=1.0).contains(&current_value) {
            return Err(EvaluationError::new("invalid inference value"));
        }
        let mut policy = Vec::with_capacity(policy_size);
        for _ in 0..policy_size {
            let logit = read_f32(&mut reader)?;
            if !logit.is_finite() {
                return Err(EvaluationError::new("non-finite policy logit"));
            }
            policy.push(logit);
        }
        let output = InferenceOutput::new(id, current_value, policy)?;
        if outputs.insert(id, output).is_some() {
            return Err(EvaluationError::new("duplicate inference response ID"));
        }
    }
    batch
        .iter()
        .map(|input| {
            outputs
                .remove(&input.id)
                .ok_or_else(|| EvaluationError::new("missing inference response ID"))
        })
        .collect()
}

fn push_u32(buffer: &mut Vec<u8>, value: u32) {
    buffer.extend_from_slice(&value.to_le_bytes());
}

fn read_u32(reader: &mut impl Read) -> Result<u32, EvaluationError> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes).map_err(io_error)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut impl Read) -> Result<u64, EvaluationError> {
    let mut bytes = [0_u8; 8];
    reader.read_exact(&mut bytes).map_err(io_error)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_f32(reader: &mut impl Read) -> Result<f32, EvaluationError> {
    let mut bytes = [0_u8; 4];
    reader.read_exact(&mut bytes).map_err(io_error)?;
    Ok(f32::from_le_bytes(bytes))
}

fn io_error(error: std::io::Error) -> EvaluationError {
    EvaluationError::new(format!("inference transport: {error}"))
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use vgo_core::{Color, Position};
    use vgo_raster::{RasterConfig, rasterize};

    use super::{
        InferenceInput, PROTOCOL_VERSION, RESPONSE_MAGIC, encode_request_frame, read_response_frame,
    };

    fn inputs() -> Vec<InferenceInput> {
        let position = Position::new(0.1, Vec::new(), Color::Black);
        let raster = rasterize(&position, RasterConfig::square(2));
        vec![
            InferenceInput::new(10, raster.clone()),
            InferenceInput::new(20, raster),
        ]
    }

    #[test]
    fn request_frame_is_contiguous_and_versioned() {
        let frame = encode_request_frame(&inputs()).expect("valid request frame");
        assert_eq!(&frame[..8], b"VGOIFR01");
        assert_eq!(u32::from_le_bytes(frame[8..12].try_into().unwrap()), 1);
        assert_eq!(u32::from_le_bytes(frame[12..16].try_into().unwrap()), 2);
        assert_eq!(frame.len(), 28 + 2 * (8 + 10 * 2 * 2 * 4));
    }

    #[test]
    fn responses_are_routed_by_identifier() {
        let inputs = inputs();
        let policy_size = 5_u32;
        let mut response = Vec::new();
        response.extend_from_slice(&RESPONSE_MAGIC);
        response.extend_from_slice(&PROTOCOL_VERSION.to_le_bytes());
        response.extend_from_slice(&2_u32.to_le_bytes());
        response.extend_from_slice(&policy_size.to_le_bytes());
        for (id, value, logit) in [(20_u64, -0.25_f32, 2.0_f32), (10, 0.5, 1.0)] {
            response.extend_from_slice(&id.to_le_bytes());
            response.extend_from_slice(&value.to_le_bytes());
            for _ in 0..policy_size {
                response.extend_from_slice(&logit.to_le_bytes());
            }
        }
        let outputs = read_response_frame(&mut Cursor::new(response), &inputs)
            .expect("valid framed response");
        assert_eq!(outputs[0].id(), 10);
        assert_eq!(outputs[0].current_value(), 0.5);
        assert_eq!(outputs[0].policy(), vec![1.0; policy_size as usize]);
        assert_eq!(outputs[1].id(), 20);
        assert_eq!(outputs[1].current_value(), -0.25);
    }
}
