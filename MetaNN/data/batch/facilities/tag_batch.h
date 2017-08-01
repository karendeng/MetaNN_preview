#pragma once

#include <MetaNN/data/facilities/tags.h>
#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/data/batch/batch.h>
#include <MetaNN/evaluate/facilities/eval_buffer.h>
#include <MetaNN/evaluate/facilities/eval_group.h>
#include <MetaNN/evaluate/facilities/eval_plan.h>
#include <MetaNN/evaluate/facilities/eval_unit.h>
#include <cassert>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace MetaNN
{
namespace NSBatchMatrix
{
template <typename TInputHandle, typename TOutputHandle, typename TDevice>
class EvalUnit;

template <typename TInputHandle, typename TOutputHandle>
class EvalUnit<TInputHandle, TOutputHandle, DeviceTags::CPU>
    : public BaseEvalUnit<DeviceTags::CPU>
{
public:
    EvalUnit(std::vector<TInputHandle> opers,
             TOutputHandle evalRes)
        : m_opers(std::move(opers))
        , m_evalRes(std::move(evalRes)) {}

    void Eval() override
    {
        auto& res = m_evalRes.MutableData();
        assert(!m_evalRes.IsEvaluated());
        res.Reserve(m_opers.size());
        for (size_t i = 0; i < m_opers.size(); ++i)
        {
            res.PushBack(m_opers[i].Data());
        }
        m_evalRes.SetEval();
    }

    size_t OperandDepth(const std::unordered_map<const void*, size_t>& opMap) const
    {
        int res = -1;
        for (const auto& h : m_opers)
        {
            auto it = opMap.find(h.DataPtr());
            if (it != opMap.end())
            {
                res = std::max(res, (int)(it->second));
            }
        }
        return (size_t)res;
    }

private:
    std::vector<TInputHandle> m_opers;
    TOutputHandle m_evalRes;
};
}

template <typename TData, typename TDataCate>
class TagBatch;

template <typename TTagBatch>
class BatchEvalRegister;

template <typename TData>
class BatchEvalRegister<TagBatch<TData, CategoryTags::Matrix>>
{
public:
    using ElementType = typename TData::ElementType;
    using DeviceType = typename TData::DeviceType;
    
protected:
    bool ModifyAvail() const
    {
        return !(m_evalBuf.IsEvaluated());
    }
    
public:
    auto EvalRegister() const
    {
        using TTagBatch = TagBatch<TData, CategoryTags::Matrix>;
        const auto& tmp = static_cast<const TTagBatch&>(*this);
        if (!(m_evalBuf.IsEvaluated()))
        {
            using TOpEvalHandle = std::decay_t<decltype(std::declval<TData>().EvalRegister())>;
            std::vector<TOpEvalHandle> handleBuf;
            handleBuf.reserve(tmp.BatchNum());
            for (size_t i = 0; i < tmp.BatchNum(); ++i)
            {
                handleBuf.push_back(tmp[i].EvalRegister());
            }

            auto outHandle = m_evalBuf.Handle();
            
            using EvalUnit = NSBatchMatrix::EvalUnit<TOpEvalHandle, decltype(outHandle), DeviceType>;
            using GroupType = TrivalEvalGroup<EvalUnit>;
            
            outHandle.Allocate(tmp.RowNum(), tmp.ColNum());
            const void* dataPtr = outHandle.DataPtr();
            EvalUnit unit(std::move(handleBuf), std::move(outHandle));
            
            EvalPlan<DeviceType>::template Register<GroupType>(std::move(unit), dataPtr);
        }
        return m_evalBuf.ConstHandle();
    }
    
private:
    EvalBuffer<Batch<Matrix<ElementType, DeviceType>>> m_evalBuf;
};

template <typename TElem, typename TDevice>
class BatchEvalRegister<TagBatch<Matrix<TElem, TDevice>, CategoryTags::Matrix>>
{
protected:
    bool ModifyAvail() const
    {
        return true;
    }
    
public:
    auto EvalRegister() const
    {
        const auto& tmp = static_cast<const Batch<Matrix<TElem, TDevice>>&>(*this);
        return MakeConstEvalHandle(tmp);
    }
};

template <typename TData>
class TagBatch<TData, CategoryTags::Matrix>
    : public BatchEvalRegister<TagBatch<TData, CategoryTags::Matrix>>
{
    using TBase = BatchEvalRegister<TagBatch<TData, CategoryTags::Matrix>>;
public:
    using ElementType = typename TData::ElementType;
    using DeviceType = typename TData::DeviceType;

    TagBatch(size_t rowNum = 0, size_t colNum = 0)
        : m_rowNum(rowNum)
        , m_colNum(colNum) {}
        
    template <typename TIterator>
    TagBatch(size_t rowNum, size_t colNum,
              TIterator b, TIterator e)
        : m_rowNum(rowNum)
        , m_colNum(colNum)
    {
        for (auto& p : m_buffer)
        {
            if ((p.RowNum() != rowNum) || (p.ColNum() != colNum))
            {
                throw std::runtime_error("Dimension mismatch");
            }
        }
    }
    
public:
    size_t RowNum() const { return m_rowNum; }
    size_t ColNum() const { return m_colNum; }

    void PushBack(TData mat)
    {
        assert(TBase::ModifyAvail());
        if ((mat.RowNum() != m_rowNum) || (mat.ColNum() != m_colNum))
        {
            throw std::runtime_error("Dimension mismatch");
        }
        m_buffer.emplace_back(std::move(mat));
    }
    
    template <typename...TArgs>
    void EmplaceBack(TArgs&&... args)
    {
        assert(TBase::ModifyAvail());
        TData tmp(std::forward<TArgs>(args)...);
        if ((tmp.RowNum() != m_rowNum) || (tmp.ColNum() != m_colNum))
        {
            throw std::runtime_error("Dimension mismatch");
        }
        m_buffer.emplace_back(std::move(tmp));
    }
    
    size_t BatchNum() const
    {
        return m_buffer.size();
    }
    
    void Reserve(size_t num)
    {
        assert(TBase::ModifyAvail());
        m_buffer.reserve(num);
    }
    
    void Clear()
    {
        assert(TBase::ModifyAvail());
        m_buffer.clear();
    }
    
    bool IsEmpty() const
    {
        return m_buffer.empty();
    }
    
    const auto& operator[] (size_t id) const
    {
        return m_buffer[id];
    }
    
    auto& operator[] (size_t id)
    {
        return m_buffer[id];
    }
    
    auto begin() { return m_buffer.begin(); }
    auto begin() const { return m_buffer.begin(); }
    auto end() { return m_buffer.end(); }
    auto end() const { return m_buffer.end(); }
    
    template <typename TBatch,
              std::enable_if_t<IsBatchMatrix<TBatch>>* = nullptr>
    bool operator== (const TBatch& val) const
    {
        auto tmp = static_cast<const TagBatch&>(val);
        return m_buffer == tmp.m_buffer;
    }

    template <typename TOtherData,
              std::enable_if_t<!IsBatchMatrix<TOtherData>>* = nullptr>
    bool operator== (const TOtherData&) const
    {
        return false;
    }

    template <typename TOtherData>
    bool operator!= (const TOtherData& val) const
    {
        return !(operator==(val));
    }
    
protected:
    size_t m_rowNum;
    size_t m_colNum;
    std::vector<TData> m_buffer;
};
}