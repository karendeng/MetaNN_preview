#pragma once

#include <MetaNN/data/facilities/tags.h>
#include <MetaNN/data/matrices/matrices.h>
#include <MetaNN/evaluate/facilities/eval_buffer.h>
#include <MetaNN/evaluate/facilities/eval_group.h>
#include <MetaNN/evaluate/facilities/eval_handle.h>
#include <MetaNN/evaluate/facilities/eval_plan.h>
#include <MetaNN/evaluate/facilities/eval_unit.h>
#include <cassert>
#include <stdexcept>

namespace MetaNN
{
namespace NSZeroMatrix
{
template <typename TElement, typename TDevice>
class EvalUnit;

template <typename TElement>
class EvalUnit<TElement, DeviceTags::CPU> : public BaseEvalUnit<DeviceTags::CPU>
{
public:
    using ElementType = TElement;

    EvalUnit(EvalHandle<Matrix<TElement, DeviceTags::CPU>> resBuf,
             size_t rowNum, size_t colNum)
        : m_resHandle(std::move(resBuf))
        , m_rowNum(rowNum)
        , m_colNum(colNum) {}

    void Eval() override
    {
        m_resHandle.Allocate(m_rowNum, m_colNum);
        auto lowLayer = LowerAccess(m_resHandle.Data());
        const size_t rowLen = lowLayer.RowLen();
        auto mem = lowLayer.MutableRawMemory();
        if (rowLen != m_colNum)
        {
            throw std::runtime_error("Gap among matrix rows");
        }
        memset(mem, 0, sizeof(TElement) * m_colNum * m_rowNum);
    }

    size_t OperandDepth(const std::unordered_map<const void*, size_t>&) const
    {
        return (size_t)-1;
    }
    
private:
    EvalHandle<Matrix<TElement, DeviceTags::CPU>> m_resHandle;
    size_t m_rowNum;
    size_t m_colNum;
};
}

template<typename TElem, typename TDevice>
class ZeroMatrix
{
    static_assert(std::is_same<std::decay_t<TElem>, TElem>::value,
                  "TElem is not an available type");
public:
    using DeviceType = TDevice;
    using ElementType = TElem;

public:
    ZeroMatrix(size_t p_rowNum, size_t p_colNum)
        : m_rowNum(p_rowNum)
        , m_colNum(p_colNum) {}

    bool operator== (const ZeroMatrix& val) const
    {
        return (m_rowNum == val.m_rowNum) &&
               (m_colNum == val.m_colNum);
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

    auto EvalRegister() const
    {
        using TEvalUnit = NSZeroMatrix::EvalUnit<ElementType, DeviceType>;
        using TEvalGroup = TrivalEvalGroup<TEvalUnit>;
        if (m_evalBuf.IsEmpty())
        {
            auto evalHandle = m_evalBuf.Handle();
            TEvalUnit unit(evalHandle, m_rowNum, m_colNum);
            EvalPlan<DeviceType>::template Register<TEvalGroup>(unit, evalHandle.DataPtr());
        }
        return m_evalBuf.ConstHandle();
    }
    
private:
    size_t m_rowNum;
    size_t m_colNum;
    EvalBuffer<Matrix<TElem, TDevice>> m_evalBuf;
};

template <typename TElem, typename TDevice>
constexpr bool IsMatrix<ZeroMatrix<TElem, TDevice>> = true;
}