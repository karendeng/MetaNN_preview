#pragma once

#include <MetaNN/data/facilities/traits.h>
#include <MetaNN/data/matrices/trival_matrix.h>
#include <MetaNN/evaluate/facilities/eval_plan.h>
#include <MetaNN/operators/facilities/tags.h>
#include <MetaNN/operators/operators.h>
#include <cassert>
#include <type_traits>
#include <utility>

namespace MetaNN
{
namespace NSAbs
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
        const size_t rowNum = p_v.RowNum();
        const size_t colNum = p_v.ColNum();
        
        m_evalOutput.Allocate(rowNum, colNum);
        auto& res = m_evalOutput.Data();
        
        auto mem_v1 = LowerAccess(p_v);
        auto mem_res = LowerAccess(res);

        const size_t src1PackNum = mem_v1.RowLen();
        const size_t tgtPackNum = mem_res.RowLen();

        const ElementType* r1 = mem_v1.RawMemory();
        ElementType* r = mem_res.MutableRawMemory();

        constexpr auto zeroValue = ElementType();
        for (size_t i = 0; i < rowNum; ++i)
        {
            for (size_t j = 0; j < colNum; ++j)
            {
                r[j] = (r1[j] > zeroValue) ? r1[j] : -r1[j];
            }
            r1 += src1PackNum;
            r += tgtPackNum;
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
struct OperBuildInSeq_<UnaryOpTags::Abs>
{
    using type = OperSeqContainer<NSAbs::GeneralCase>;
};

template <typename TP>
struct OperAbs_
{
// valid check
private:
    using rawM = std::decay_t<TP>;

public:
    static constexpr bool valid = IsMatrix<rawM>;

public:
    static auto Eval(TP&& p_m)
    {
        using ResType = UnaryOp<UnaryOpTags::Abs, rawM>;
        return ResType(std::forward<TP>(p_m));
    }
};

template <typename TP,
          std::enable_if_t<OperAbs_<TP>::valid>* = nullptr>
auto Abs(TP&& p_m)
{
    return OperAbs_<TP>::Eval(std::forward<TP>(p_m));
}
}
