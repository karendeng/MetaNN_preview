#pragma once

namespace MetaNN
{
template <typename T>
class Batch;

template <typename T>
constexpr bool IsBatchMatrix<Batch<T>> = IsMatrix<T>;
}