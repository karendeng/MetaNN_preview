#pragma once

#include <MetaNN/data/facilities/tags.h>
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
namespace NSTrivalMatrix
{
template <typename TElem, typename TDevice>
class EvalUnit;

template <typename TElem>
class EvalUnit<TElem, DeviceTags::CPU>
    : public BaseEvalUnit<DeviceTags::CPU>
{
    using ScalarType = Scalar<TElem, DeviceTags::CPU>;
    using StorageType = typename ScalarType::StorageType;
public:
    EvalUnit(EvalHandle<Matrix<TElem, DeviceTags::CPU>> resBuf,
             size_t rowNum, size_t colNum, StorageType val)
        : m_resHandle(std::move(resBuf))
        , m_rowNum(rowNum)
        , m_colNum(colNum)
        , m_val(val) {}

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
                throw std::runtime_error("Row number mismatch for TrivalMatrix::Eval");
            }
            if (mutableData.ColNum() != m_colNum)
            {
                throw std::runtime_error("Column number mismatch for TrivalMatrix::Eval");
            }
        }
        auto lowLayer = LowerAccess(mutableData);
        const size_t rowLen = lowLayer.RowLen();
        auto mem = lowLayer.MutableRawMemory();
        for (size_t i = 0; i < m_rowNum; ++i)
        {
            for (size_t j = 0; j < m_colNum; ++j)
            {
                mem[j] = m_val;
            }
            mem += rowLen;
        }
        m_resHandle.SetEval();
    }

    size_t OperandDepth(const std::unordered_map<const void*, size_t>&) const
    {
        return (size_t)-1;
    }
    
private:
    EvalHandle<Matrix<TElem, DeviceTags::CPU>> m_resHandle;
    size_t m_rowNum;
    size_t m_colNum;
    StorageType m_val;
};
}

template<typename TElem, typename TDevice>
class TrivalMatrix
{
public:
    using ElementType = TElem;
    using DeviceType = TDevice;
    
private:
    using StorageType = typename Scalar<ElementType, DeviceType>::StorageType;
    
public:
    TrivalMatrix(size_t p_rowNum, size_t p_colNum,
                 StorageType p_val)
        : m_val(p_val)
        , m_rowNum(p_rowNum)
        , m_colNum(p_colNum) {}

    bool operator== (const TrivalMatrix& val) const
    {
        return (m_val == val.m_val) &&
               (m_rowNum == val.m_rowNum) &&
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

    size_t RowNum() const
    {
        return m_rowNum;
    }

    size_t ColNum() const
    {
        return m_colNum;
    }

    auto EvalRegister() const
    {
        using TEvalUnit = NSTrivalMatrix::EvalUnit<ElementType, DeviceType>;
        using TEvalGroup = TrivalEvalGroup<TEvalUnit>;
        if (!m_evalBuf.IsEvaluated())
        {
            auto evalHandle = m_evalBuf.Handle();
            TEvalUnit unit(evalHandle, m_rowNum, m_colNum, m_val);
            EvalPlan<DeviceType>::template Register<TEvalGroup>(unit, evalHandle.DataPtr());
        }
        return m_evalBuf.ConstHandle();
    }

    auto ElementValue() const
    {
        return m_val;
    }

private:
    StorageType m_val;
    size_t m_rowNum;
    size_t m_colNum;
    EvalBuffer<Matrix<ElementType, DeviceType>> m_evalBuf;
};

template <typename TElem, typename TDevice>
constexpr bool IsMatrix<TrivalMatrix<TElem, TDevice>> = true;
}
