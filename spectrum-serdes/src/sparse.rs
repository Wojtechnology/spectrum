use std::io::{Read, Write};
use std::marker::PhantomData;

use protobuf::{CodedInputStream, Message};

use crate::error::{ReadError, WriteError};
use crate::model::sparse_stream::{SparseStream, SparseStreamMeta};

mod inner {
    // We don't want to expse these publically, to prevent outside implementations.
    use protobuf::RepeatedField;

    use crate::model::sparse_stream::{
        F32SparsePoint, F32SparseStream, F64SparsePoint, F64SparseStream, SparseStream,
    };

    pub struct SparsePoint<T> {
        pub index: u64,
        pub points: Vec<T>,
    }

    pub trait Sparseable {
        type T;

        fn set_points(stream: &mut SparseStream, points: &Vec<SparsePoint<Self::T>>);

        fn get_point(stream: &SparseStream, i: u64) -> Option<SparsePoint<Self::T>>;
    }

    impl Sparseable for f32 {
        type T = f32;

        fn set_points(stream: &mut SparseStream, points: &Vec<SparsePoint<f32>>) {
            let mut sparse_points = Vec::new();
            for point in points.iter() {
                let mut sparse_point = F32SparsePoint::new();
                sparse_point.set_index(point.index);
                sparse_point.set_values(point.points.clone());
                sparse_points.push(sparse_point);
            }
            let mut f32_stream = F32SparseStream::new();
            f32_stream.set_points(RepeatedField::from_vec(sparse_points));
            stream.set_f32_stream(f32_stream);
        }

        fn get_point(stream: &SparseStream, i: u64) -> Option<SparsePoint<f32>> {
            assert!(stream.has_f32_stream(), "Must have f32 stream");
            let points = stream.get_f32_stream().get_points();
            if (i as usize) < points.len() {
                Some(SparsePoint {
                    index: points[i as usize].get_index(),
                    points: points[i as usize].get_values().to_vec(),
                })
            } else {
                None
            }
        }
    }

    impl Sparseable for f64 {
        type T = f64;

        fn set_points(stream: &mut SparseStream, points: &Vec<SparsePoint<f64>>) {
            let mut sparse_points = Vec::new();
            for point in points.iter() {
                let mut sparse_point = F64SparsePoint::new();
                sparse_point.set_index(point.index);
                sparse_point.set_values(point.points.clone());
                sparse_points.push(sparse_point);
            }
            let mut f64_stream = F64SparseStream::new();
            f64_stream.set_points(RepeatedField::from_vec(sparse_points));
            stream.set_f64_stream(f64_stream);
        }

        fn get_point(stream: &SparseStream, i: u64) -> Option<SparsePoint<f64>> {
            assert!(stream.has_f64_stream(), "Must have f64 stream");
            let points = stream.get_f64_stream().get_points();
            if (i as usize) < points.len() {
                Some(SparsePoint {
                    index: points[i as usize].get_index(),
                    points: points[i as usize].get_values().to_vec(),
                })
            } else {
                None
            }
        }
    }
}

pub struct SparseStreamWriter<T: inner::Sparseable<T = T>> {
    buffer: Vec<inner::SparsePoint<T>>,
    channels: u32,
    sample_rate: f64,
}

impl<T: inner::Sparseable<T = T>> SparseStreamWriter<T> {
    pub fn new(channels: u32, sample_rate: f64) -> Self {
        Self {
            buffer: Vec::new(),
            channels,
            sample_rate,
        }
    }

    pub fn push(&mut self, index: u64, points: Vec<T>) {
        assert!(
            self.buffer.len() == 0 || self.buffer[self.buffer.len() - 1].index < index,
            format!(
                "Sparse point indices must be increasing ({} >= {})",
                self.buffer[self.buffer.len() - 1].index,
                index
            ),
        );
        assert!(
            points.len() as u32 == self.channels,
            format!(
                "Number of points must be same as channels ({} != {})",
                points.len(),
                self.channels
            )
        );

        self.buffer.push(inner::SparsePoint { index, points });
    }

    pub fn write(&self, writer: &mut dyn Write) -> Result<(), WriteError> {
        let mut meta = SparseStreamMeta::new();
        meta.set_size(self.buffer.len() as u64);
        meta.set_channels(self.channels);
        meta.set_sample_rate(self.sample_rate);

        let mut stream = SparseStream::new();
        stream.set_meta(meta);
        T::set_points(&mut stream, &self.buffer);

        match stream.write_to_writer(writer) {
            Ok(()) => Ok(()),
            Err(e) => Err(WriteError::new(&format!("{:?}", e))),
        }
    }
}

impl SparseStream {
    pub fn from_reader(reader: &mut dyn Read) -> Result<Self, ReadError> {
        let mut data = SparseStream::new();
        match data.merge_from(&mut CodedInputStream::new(reader)) {
            Ok(()) => Ok(data),
            Err(e) => Err(ReadError::new(&format!("{:?}", e))),
        }
    }

    pub fn to_f32_iter<'a>(&'a self) -> SparseStreamReader<'a, f32> {
        assert!(self.has_f32_stream(), "Must have f32 stream");
        SparseStreamReader::new(self)
    }

    pub fn to_f64_iter<'a>(&'a self) -> SparseStreamReader<'a, f64> {
        assert!(self.has_f64_stream(), "Must have f64 stream");
        SparseStreamReader::new(self)
    }
}

pub struct SparseStreamReader<'a, T: inner::Sparseable<T = T>> {
    stream: &'a SparseStream,
    cursor: u64,
    phantom: PhantomData<T>,
}

impl<'a, T: inner::Sparseable<T = T>> SparseStreamReader<'a, T> {
    pub fn new(stream: &'a SparseStream) -> Self {
        Self {
            stream,
            cursor: 0,
            phantom: PhantomData,
        }
    }
}

impl<'a, T: inner::Sparseable<T = T>> Iterator for SparseStreamReader<'a, T> {
    type Item = (u64, Vec<T>);

    fn next(&mut self) -> Option<(u64, Vec<T>)> {
        match T::get_point(&mut self.stream, self.cursor) {
            Some(v) => {
                self.cursor += 1;
                Some((v.index, v.points))
            }
            None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_writes_and_reads_f32() {
        let mut buf = Vec::<u8>::new();
        let mut writer = SparseStreamWriter::new(2, 44100.0);
        writer.push(0, vec![123.0f32, 456.0]);
        writer.push(1, vec![234.0f32, 345.0]);
        let write_res = writer.write(&mut buf);
        write_res.expect("writer.write should be ok");

        let stream = SparseStream::from_reader(&mut &buf[..]).expect("should be able to read");
        let meta = stream.get_meta();
        assert_eq!(meta.get_size(), 2);
        assert_eq!(meta.get_channels(), 2);
        assert_eq!(meta.get_sample_rate(), 44100.0);
        assert!(!stream.has_f64_stream(), "must not have f64 channels");

        let actual: Vec<_> = stream.to_f32_iter().collect();
        let expected = vec![(0, vec![123.0f32, 456.0]), (1, vec![234.0f32, 345.0])];
        assert_eq!(actual, expected);
    }

    #[test]
    fn it_writes_and_reads_f64() {
        let mut buf = Vec::<u8>::new();
        let mut writer = SparseStreamWriter::new(2, 44100.0);
        writer.push(1, vec![123.0f64, 456.0]);
        writer.push(4, vec![234.0f64, 345.0]);
        let write_res = writer.write(&mut buf);
        write_res.expect("writer.write should be ok");

        let stream = SparseStream::from_reader(&mut &buf[..]).expect("should be able to read");
        let meta = stream.get_meta();
        assert_eq!(meta.get_size(), 2);
        assert_eq!(meta.get_channels(), 2);
        assert_eq!(meta.get_sample_rate(), 44100.0);
        assert!(!stream.has_f32_stream(), "must not have f32 channels");

        let actual: Vec<_> = stream.to_f64_iter().collect();
        let expected = vec![(1, vec![123.0f64, 456.0]), (4, vec![234.0f64, 345.0])];
        assert_eq!(actual, expected);
    }

    #[test]
    #[should_panic]
    fn it_should_panic_when_pushing_out_of_order_f32() {
        let mut writer = SparseStreamWriter::new(2, 44100.0);
        writer.push(5, vec![123.0f32, 456.0]);
        writer.push(4, vec![234.0f32, 345.0]);
    }

    #[test]
    #[should_panic]
    fn it_should_panic_when_pushing_out_of_order_f64() {
        let mut writer = SparseStreamWriter::new(2, 44100.0);
        writer.push(5, vec![123.0f64, 456.0]);
        writer.push(4, vec![234.0f64, 345.0]);
    }

    #[test]
    #[should_panic]
    fn it_should_panic_when_pushing_duplicate_f32() {
        let mut writer = SparseStreamWriter::new(2, 44100.0);
        writer.push(5, vec![123.0f32, 456.0]);
        writer.push(5, vec![234.0f32, 345.0]);
    }

    #[test]
    #[should_panic]
    fn it_should_panic_when_pushing_duplicate_f64() {
        let mut writer = SparseStreamWriter::new(2, 44100.0);
        writer.push(5, vec![123.0f64, 456.0]);
        writer.push(5, vec![234.0f64, 345.0]);
    }

    #[test]
    #[should_panic]
    fn it_should_panic_when_iterating_wrong_type_f32() {
        let mut buf = Vec::<u8>::new();
        let mut writer = SparseStreamWriter::new(2, 44100.0);
        writer.push(0, vec![123.0f32, 456.0]);
        writer.push(1, vec![234.0f32, 345.0]);
        let write_res = writer.write(&mut buf);
        write_res.expect("writer.write should be ok");

        let stream = SparseStream::from_reader(&mut &buf[..]).expect("should be able to read");
        stream.to_f64_iter();
    }

    #[test]
    #[should_panic]
    fn it_should_panic_when_iterating_wrong_type_f64() {
        let mut buf = Vec::<u8>::new();
        let mut writer = SparseStreamWriter::new(2, 44100.0);
        writer.push(0, vec![123.0f64, 456.0]);
        writer.push(1, vec![234.0f64, 345.0]);
        let write_res = writer.write(&mut buf);
        write_res.expect("writer.write should be ok");

        let stream = SparseStream::from_reader(&mut &buf[..]).expect("should be able to read");
        stream.to_f32_iter();
    }
}
