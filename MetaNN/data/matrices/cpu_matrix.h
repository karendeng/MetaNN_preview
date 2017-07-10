#pragma once

#include <MetaNN/data/facilities/continuous_memory.h>
#include <MetaNN/data/facilities/lower_access.h>
#include <MetaNN/data/matrices/matrices.h>
#include <MetaNN/evaluate/facilities/eval_handle.h>
#include <cassert>
#include <cstring>
#include <type_traits>

namespace MetaNN
{
template <typename TElem>
struct LowerAccessImpl<Matrix<TElem, DeviceTags::CPU>>;

template <typename TElem>
class Matrix<TElem, DeviceTags::CPU>
{
    static_assert(std::is_same<std::decay_t<TElem>, TElem>::value,
                  "TElem is not an available type");
public:
    using DeviceType = DeviceTags::CPU;
    using ElementType = TElem;

    friend struct LowerAccessImpl<Matrix<ElementType, DeviceTags::CPU>>;

public:
    explicit Matrix()
        : Matrix(0, 0) {}

    explicit Matrix(size_t p_rowNum, size_t p_colNum)
        : m_mem(p_rowNum * p_colNum)
        , m_rowNum(p_rowNum)
        , m_colNum(p_colNum)
        , m_rowLen(p_colNum)
    {}

    bool operator== (const Matrix& val) const
    {
        return (m_mem == val.m_mem) &&
               (m_rowNum == val.m_rowNum) &&
               (m_colNum == val.m_colNum) &&
               (m_rowLen == val.m_rowLen);
    }

    template <typename TOtherType>
    bool operator== (const TOtherType&) const
    {
        return false;
    }

    template <typename TData>
    bool operator!= (const TData& val) const
    {
        return !(operator==(val));
    }

    size_t RowNum() const { return m_rowNum; }
    size_t ColNum() const { return m_colNum; }

    bool AvailableForWrite() const { return m_mem.UseCount() == 1; }

    void SetValue(size_t p_rowId, size_t p_colId, ElementType val)
    {
        assert(AvailableForWrite());
        assert((p_rowId < m_rowNum) && (p_colId < m_colNum));
        (m_mem.RawMemory())[p_rowId * m_rowLen + p_colId] = val;
    }

    const TElem operator () (size_t p_rowId, size_t p_colId) const
    {
        assert((p_rowId < m_rowNum) && (p_colId < m_colNum));
        return (m_mem.RawMemory())[p_rowId * m_rowLen + p_colId];
    }

    Matrix SubMatrix(size_t p_rowB, size_t p_rowE, size_t p_colB, size_t p_colE) const
    {
        assert((p_rowB < m_rowNum) && (p_colB < m_colNum));
        assert((p_rowE <= m_rowNum) && (p_colE <= m_colNum));
        TElem* pos = m_mem.RawMemory() + p_rowB * m_rowLen + p_colB;
        return Matrix(m_mem.SharedPtr(), pos,
                      p_rowE - p_rowB, p_colE - p_colB,
                      m_rowLen);
    }

    auto EvalRegister() const
    {
        return MakeConstEvalHandle(*this);
    }

private:
    Matrix(std::shared_ptr<TElem> p_mem,
            TElem* p_memStart,
            size_t p_rowNum,
            size_t p_colNum,
            size_t p_rowLen)
        : m_mem(p_mem, p_memStart)
        , m_rowNum(p_rowNum)
        , m_colNum(p_colNum)
        , m_rowLen(p_rowLen)
    {}

private:
    ContinuousMemory<TElem, DeviceType> m_mem;
    size_t m_rowNum;
    size_t m_colNum;
    size_t m_rowLen;
};

template<typename TElem>
struct LowerAccessImpl<Matrix<TElem, DeviceTags::CPU>>
{
    LowerAccessImpl(Matrix<TElem, DeviceTags::CPU>& p)
        : m_matrix(p)
    {}

    TElem* MutableRawMemory()
    {
        assert(m_matrix.AvailableForWrite());
        return m_matrix.m_mem.RawMemory();
    }

    const TElem* RawMemory() const
    {
        return m_matrix.m_mem.RawMemory();
    }

    size_t RowLen() const
    {
        return m_matrix.m_rowLen;
    }

private:
    Matrix<TElem, DeviceTags::CPU>& m_matrix;
};
}
