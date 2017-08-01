#pragma once
#include <MetaNN/data/facilities/tags.h>

namespace MetaNN
{
template <typename TElem, typename TDevice> class Matrix;
template <typename TElem, typename TDevice> class Scalar;
template <typename TData> class Batch;

template <typename TCategory, typename TElem, typename TDevice>
struct PrincipalDataType_;

template <typename TElem, typename TDevice>
struct PrincipalDataType_<CategoryTags::Scalar, TElem, TDevice>
{
    using type = Scalar<TElem, TDevice>;
};

template <typename TElem, typename TDevice>
struct PrincipalDataType_<CategoryTags::Matrix, TElem, TDevice>
{
    using type = Matrix<TElem, TDevice>;
};

template <typename TElem, typename TDevice>
struct PrincipalDataType_<CategoryTags::BatchMatrix, TElem, TDevice>
{
    using type = Batch<Matrix<TElem, TDevice>>;
};

template <typename TCategory, typename TElem, typename TDevice>
using PrincipalDataType = typename PrincipalDataType_<TCategory, TElem, TDevice>::type;

/// is scalar
template <typename T>
constexpr bool IsScalar = false;

template <typename T>
constexpr bool IsScalar<const T> = IsScalar<T>;

template <typename T>
constexpr bool IsScalar<T&> = IsScalar<T>;

template <typename T>
constexpr bool IsScalar<T&&> = IsScalar<T>;

/// is matrix
template <typename T>
constexpr bool IsMatrix = false;

template <typename T>
constexpr bool IsMatrix<const T> = IsMatrix<T>;

template <typename T>
constexpr bool IsMatrix<T&> = IsMatrix<T>;

template <typename T>
constexpr bool IsMatrix<T&&> = IsMatrix<T>;

/// is batch matrix
template <typename T>
constexpr bool IsBatchMatrix = false;

template <typename T>
constexpr bool IsBatchMatrix<const T> = IsBatchMatrix<T>;

template <typename T>
constexpr bool IsBatchMatrix<T&> = IsBatchMatrix<T>;

template <typename T>
constexpr bool IsBatchMatrix<const T&> = IsBatchMatrix<T>;

template <typename T>
constexpr bool IsBatchMatrix<T&&> = IsBatchMatrix<T>;

template <typename T>
constexpr bool IsBatchMatrix<const T&&> = IsBatchMatrix<T>;

template <typename T>
struct DataCategory_
{
private:
    template <bool isScalar, bool isMatrix, bool isBatchMatrix, typename TDummy = void>
    struct helper;

    template <typename TDummy>
    struct helper<true, false, false, TDummy>
    {
        using type = CategoryTags::Scalar;
    };

    template <typename TDummy>
    struct helper<false, true, false, TDummy>
    {
        using type = CategoryTags::Matrix;
    };

    template <typename TDummy>
    struct helper<false, false, true, TDummy>
    {
        using type = CategoryTags::BatchMatrix;
    };

public:
    using type = typename helper<IsScalar<T>, IsMatrix<T>, IsBatchMatrix<T>>::type;
};

template <typename T>
using DataCategory = typename DataCategory_<T>::type;

template <typename T>
struct RemConstRef_
{
    using type = T;
};

template <typename T>
struct RemConstRef_<const T>
{
    using type = typename RemConstRef_<T>::type;
};

template <typename T>
struct RemConstRef_<T&&>
{
    using type = typename RemConstRef_<T>::type;
};

template <typename T>
struct RemConstRef_<T&>
{
    using type = typename RemConstRef_<T>::type;
};

template <typename T>
using RemConstRef = typename RemConstRef_<T>::type;
}
