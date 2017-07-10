#pragma once

#include <type_traits>
#include <MetaNN/operators/operators.h>
#include <cmath>
#include <algorithm>

namespace MetaNN
{
namespace NSRowSoftmax
{
template <typename TOperHandle, typename TDevice>
class EvalUnit;

template <typename TOperHandle>
class EvalUnit<TOperHandle, DeviceTags::CPU> : public BaseEvalUnit<DeviceTags::CPU>
{
    using TOperandData = std::decay_t<decltype(std::declval<TOperHandle>().Data())>;
public:
    using ElementType = typename TOperandData::ElementType;
    using DeviceType = typename TOperandData::DeviceType;
    static_assert(std::is_same<DeviceType, DeviceTags::CPU>::value,
                  "Device type mismatch");

    EvalUnit(TOperHandle oper,
             EvalHandle<Matrix<ElementType, DeviceType>> evalOutput)
        : m_oper(std::move(oper))
        , m_evalOutput(evalOutput) { }

    size_t OperandDepth(const std::unordered_map<const void*, size_t>& depMap) const
    {
        int res = -1;

        auto it = depMap.find(m_oper.DataPtr());
        if (it != depMap.end()) res = std::max(res, (int)(it->second));

        return (size_t)res;
    }
    
    void Eval() override
    {
        const auto& p_v = m_oper.Data();
        assert(p_v.RowNum() == 1);
        const size_t colNum = p_v.ColNum();
        
        m_evalOutput.Allocate(1, colNum);
        if (colNum == 0) return;
        auto& res = m_evalOutput.Data();

        auto mem_v1 = LowerAccess(p_v);
        auto mem_res = LowerAccess(res);

        const ElementType* r1 = mem_v1.RawMemory();
        ElementType* r = mem_res.MutableRawMemory();
        
        auto maxElem = *std::max_element(r1, r1 + colNum);

        ElementType sum = ElementType();

        for (size_t i = 0; i < colNum; ++i)
        {
            r[i] = exp(r1[i] - maxElem);
            sum += r[i];
        }

        for (size_t i = 0; i < colNum; ++i)
        {
            r[i] /= sum;
        }
    }

private:
    TOperHandle m_oper;
    EvalHandle<Matrix<ElementType, DeviceType>> m_evalOutput;
};

struct GeneralCase
{
    template <typename TCaseTail, typename TEvalRes, typename TOperand>
    static void EvalRegister(TEvalRes& evalRes, const TOperand& oper)
    {
        static_assert(std::is_same<TCaseTail, OperSeqContainer<>>::value,
                      "General Case is not the last one");
                      
        using RawEvalRes = typename TEvalRes::DataType;
        using DeviceType = typename RawEvalRes::DeviceType;

        auto handle = oper.EvalRegister();
        using UnitType = EvalUnit<decltype(handle), DeviceType>;
        using GroupType = TrivalEvalGroup<UnitType>;

        auto outHandle = evalRes.Handle();
        const void* dataPtr = outHandle.DataPtr();
        UnitType unit(std::move(handle), std::move(outHandle));
        EvalPlan<DeviceType>::template Register<GroupType>(std::move(unit), dataPtr);
    }
};
}

template <>
struct OperBuildInSeq_<UnaryOpTags::RowSoftmax>
{
    using type = OperSeqContainer<NSRowSoftmax::GeneralCase>;
};

template <typename TP>
struct OperRowSoftmax_
{
// valid check
private:
    using rawM = std::decay_t<TP>;

public:
    static constexpr bool valid = IsMatrix<rawM>;

public:
    template <typename T>
    static auto Eval(TP&& p_m)
    {
        using ResType = UnaryOp<UnaryOpTags::RowSoftmax, rawM>;
        return ResType(std::forward<TP>(p_m));
    }
};

template <typename TP,
          std::enable_if_t<OperRowSoftmax_<TP>::valid>* = nullptr>
auto RowSoftmax(TP&& p_m)
{
    return OperRowSoftmax_<TP>::
            template Eval<DataCategory<TP>>(std::forward<TP>(p_m));
}

namespace NSColSoftmax
{
template <typename TOperHandle, typename TDevice>
class EvalUnit;

template <typename TOperHandle>
class EvalUnit<TOperHandle, DeviceTags::CPU> : public BaseEvalUnit<DeviceTags::CPU>
{
    using TOperandData = std::decay_t<decltype(std::declval<TOperHandle>().Data())>;
public:
    using ElementType = typename TOperandData::ElementType;
    using DeviceType = typename TOperandData::DeviceType;
    static_assert(std::is_same<DeviceType, DeviceTags::CPU>::value,
                  "Device type mismatch");

    EvalUnit(TOperHandle oper,
             EvalHandle<Matrix<ElementType, DeviceType>> evalOutput)
        : m_oper(std::move(oper))
        , m_evalOutput(evalOutput) { }

    size_t OperandDepth(const std::unordered_map<const void*, size_t>& depMap) const
    {
        int res = -1;

        auto it = depMap.find(m_oper.DataPtr());
        if (it != depMap.end()) res = std::max(res, (int)(it->second));

        return (size_t)res;
    }
    
    void Eval() override
    {
        const auto& p_v = m_oper.Data();
        assert(p_v.ColNum() == 1);
        const size_t rowNum = p_v.RowNum();
        
        m_evalOutput.Allocate(rowNum, 1);
        if (rowNum == 0) return;
        auto& res = m_evalOutput.Data();

        auto mem_v1 = LowerAccess(p_v);
        auto mem_res = LowerAccess(res);
        
        auto srcRowLen = mem_v1.RowLen();

        const ElementType* r1 = mem_v1.RawMemory();
        ElementType* r = mem_res.MutableRawMemory();
        
        ElementType maxElem = r1[0];
        for (size_t i = 1; i < rowNum; ++i)
        {
            auto tmp = r1[i * srcRowLen];
            maxElem = (maxElem > tmp) ? maxElem : tmp;
        }

        ElementType sum = ElementType();

        for (size_t i = 0; i < rowNum; ++i)
        {
            r[i] = exp(r1[i * srcRowLen] - maxElem);
            sum += r[i];
        }

        for (size_t i = 0; i < rowNum; ++i)
        {
            r[i] /= sum;
        }
    }

private:
    TOperHandle m_oper;
    EvalHandle<Matrix<ElementType, DeviceType>> m_evalOutput;
};

struct GeneralCase
{
    template <typename TCaseTail, typename TEvalRes, typename TOperand>
    static void EvalRegister(TEvalRes& evalRes, const TOperand& oper)
    {
        static_assert(std::is_same<TCaseTail, OperSeqContainer<>>::value,
                      "General Case is not the last one");
                      
        using RawEvalRes = typename TEvalRes::DataType;
        using DeviceType = typename RawEvalRes::DeviceType;

        auto handle = oper.EvalRegister();
        using UnitType = EvalUnit<decltype(handle), DeviceType>;
        using GroupType = TrivalEvalGroup<UnitType>;

        auto outHandle = evalRes.Handle();
        const void* dataPtr = outHandle.DataPtr();
        UnitType unit(std::move(handle), std::move(outHandle));
        EvalPlan<DeviceType>::template Register<GroupType>(std::move(unit), dataPtr);
    }
};
}

template <>
struct OperBuildInSeq_<UnaryOpTags::ColSoftmax>
{
    using type = OperSeqContainer<NSColSoftmax::GeneralCase>;
};

template <typename TP>
struct OperColSoftmax_
{
// valid check
private:
    using rawM = std::decay_t<TP>;

public:
    static constexpr bool valid = IsMatrix<rawM>;

public:
    template <typename T>
    static auto Eval(TP&& p_m)
    {
        using ResType = UnaryOp<UnaryOpTags::ColSoftmax, rawM>;
        return ResType(std::forward<TP>(p_m));
    }
};

template <typename TP,
          std::enable_if_t<OperColSoftmax_<TP>::valid>* = nullptr>
auto ColSoftmax(TP&& p_m)
{
    return OperColSoftmax_<TP>::
            template Eval<DataCategory<TP>>(std::forward<TP>(p_m));
}
}
