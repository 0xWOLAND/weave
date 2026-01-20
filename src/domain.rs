use std::ops::Range;

pub trait RangeOps {
    fn union(&self, other: &Self) -> Self;
    fn shift(&self, offset: i64) -> Self;
    fn intersection(&self, other: &Self) -> Option<Self>
    where
        Self: Sized;
}

impl RangeOps for Range<i64> {
    fn union(&self, other: &Self) -> Self {
        self.start.min(other.start)..self.end.max(other.end)
    }

    fn shift(&self, offset: i64) -> Self {
        (self.start + offset)..(self.end + offset)
    }

    fn intersection(&self, other: &Self) -> Option<Self> {
        let start = self.start.max(other.start);
        let end = self.end.min(other.end);
        (start < end).then(|| start..end)
    }
}


pub struct Domain {
    pub bounds: Vec<Option<Range<i64>>>,
}
