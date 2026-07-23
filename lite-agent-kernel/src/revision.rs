use serde::{Deserialize, Serialize};

/// An opaque compare-and-commit token supplied by a persistence backend.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct RevisionToken(Vec<u8>);

impl RevisionToken {
    pub fn initial() -> Self {
        Self::default()
    }

    pub fn from_bytes(bytes: impl Into<Vec<u8>>) -> Self {
        Self(bytes.into())
    }

    pub fn from_u64(value: u64) -> Self {
        Self(value.to_be_bytes().to_vec())
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}
