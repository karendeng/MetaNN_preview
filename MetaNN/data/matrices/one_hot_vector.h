#pragma once

#include <MetaNN/data/matrices/matrices.h>
#include <MetaNN/evaluate/facilities/eval_buffer.h>
#include <MetaNN/evaluate/facilities/eval_group.h>
#include <MetaNN/evaluate/facilities/eval_handle.h>
#include <MetaNN/evaluate/facilities/eval_plan.h>
#include <MetaNN/evaluate/facilities/eval_unit.h>

#include <cassert>
#include <memory>

namespace MetaNN
{
namespace NSOneHotVector
{
template <typename TElem, typename TDevice>
class EvalUnit;

template <typename TElement>
class EvalUnit<TElement, DeviceTags::CPU>
    : public BaseEvalUnit<DeviceTags::CPU>
{
public:
    EvalUnit(EvalHandle<Matrix<TElement, DeviceTags::CPU>> resBuf,
             size_t rowNum, size_t colNum, size_t val)
        : m_resHandle(std::move(resBuf))
        , m_rowNum(rowNum)
        , m_colNum(colNum)
        , m_val(val)
    {
        assert(m_val < (m_rowNum * m_colNum));
    }

    void Eval() override
    {
        auto& mutableData = m_resHandle.MutableData();
        if (mutableData.IsEmpty())
        {
            m_resHandle.Allocate(m_rowNum, m_colNum);
        }
        else
        {
            if (mutableData.RowNum() != m_rowNum)
            {
                throw std::runtime_error("Row number mismatch for OneHotVector::Eval");
            }
            if (mutableData.ColNum() != m_colNum)
            {
                throw std::runtime_error("Column number mismatch for OneHotVector::Eval");
            }
        }
        auto lowLayer = LowerAccess(mutableData);
        auto mem = lowLayer.MutableRawMemory();
        memset(mem, 0, sizeof(TElement) * m_rowNum * m_colNum);
        mem[m_val] = 1;
        m_resHandle.SetEval();
    }

    size_t OperandDepth(const std::unordered_map<const void*, size_t>&) const
    {
        return (size_t)-1;
    }
    
private:
    EvalHandle<Matrix<TElement, DeviceTags::CPU>> m_resHandle;
    size_t m_rowNum;
    size_t m_colNum;
    size_t m_val;
};
}

template <typename TElem, typename TDevice>
class OneHotRowVector
{
public:
    using ElementType = TElem;
    using DeviceType = TDevice;

private:
    using StorageType = typename Scalar<ElementType, DeviceType>::StorageType;

public:
    OneHotRowVector(size_t p_colNum,
                    size_t p_hotPos)
        : m_colNum(p_colNum)
        , m_hotPos(p_hotPos)
    {
        assert(p_hotPos < m_colNum);
    }

    bool operator== (const OneHotRowVector& val) const
    {
        return (m_hotPos == val.m_hotPos) &&
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

    size_t RowNum() const { return 1; }
    size_t ColNum() const { return m_colNum; }

    auto EvalRegister() const
    {
        using TEvalUnit = NSOneHotVector::EvalUnit<ElementType, DeviceType>;
        using TEvalGroup = TrivalEvalGroup<TEvalUnit>;
        if (!m_evalBuf.IsEvaluated())
        {
            auto evalHandle = m_evalBuf.Handle();
            TEvalUnit unit(evalHandle, 1, m_colNum, m_hotPos);
            EvalPlan<DeviceType>::template Register<TEvalGroup>(unit, evalHandle.DataPtr());
        }
        return m_evalBuf.ConstHandle();
    }

    auto HotPos() const
    {
        return m_hotPos;
    }

private:
    size_t m_colNum;
    size_t m_hotPos;
    EvalBuffer<Matrix<ElementType, DeviceType>> m_evalBuf;
};

template <typename TElem, typename TDevice>
constexpr bool IsMatrix<OneHotRowVector<TElem, TDevice>> = true;

template<typename TElem, typename TDevice>
class OneHotColVector
{
public:
    using ElementType = TElem;
    using DeviceType = TDevice;

private:
    using StorageType = typename Scalar<ElementType, DeviceType>::StorageType;

public:
    OneHotColVector(size_t p_rowNum,
                    size_t p_hotPos)
        : m_rowNum(p_rowNum)
        , m_hotPos(p_hotPos)
    {
        assert(p_hotPos < m_rowNum);
    }

    bool operator== (const OneHotColVector& val) const
    {
        return (m_hotPos == val.m_hotPos) &&
               (m_rowNum == val.m_rowNum);
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

    size_t ColNum() const { return 1; }

    auto EvalRegister() const
    {
        using TEvalUnit = NSOneHotVector::EvalUnit<ElementType, DeviceType>;
        using TEvalGroup = TrivalEvalGroup<TEvalUnit>;
        if (!m_evalBuf.IsEvaluated())
        {
            auto evalHandle = m_evalBuf.Handle();
            TEvalUnit unit(evalHandle, m_rowNum, 1, m_hotPos);
            EvalPlan<DeviceType>::template Register<TEvalGroup>(unit, evalHandle.DataPtr());
        }
        return m_evalBuf.ConstHandle();
    }

    auto HotPos() const
    {
        return m_hotPos;
    }

private:
    size_t m_rowNum;
    size_t m_hotPos;
    EvalBuffer<Matrix<ElementType, DeviceType>> m_evalBuf;
};
template <typename TElem, typename TDevice>
constexpr bool IsMatrix<OneHotColVector<TElem, TDevice>> = true;
}