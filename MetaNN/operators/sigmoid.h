#pragma once

#include <type_traits>
#include <MetaNN/operators/operators.h>
#include <cmath>

namespace MetaNN
{
namespace NSSigmoid
{
namespace NSCaseGen
{
template <typename TOperHandle, typename TElem, typename TDevice>
class EvalUnit;

template <typename TOperHandle, typename TElement>
class EvalUnit<TOperHandle, TElement, DeviceTags::CPU>
    : public BaseEvalUnit<DeviceTags::CPU>
{
public:
    using ElementType = TElement;
    using DeviceType = DeviceTags::CPU;

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
        auto& res = m_evalOutput.MutableData();
        
        auto mem_v1 = LowerAccess(p_v);
        auto mem_res = LowerAccess(res);

        const size_t src1PackNum = mem_v1.RowLen();
        const size_t tgtPackNum = mem_res.RowLen();
        
        using StorageType = typename Scalar<ElementType, DeviceType>::StorageType;
        const StorageType* r1 = mem_v1.RawMemory();
        StorageType* r = mem_res.MutableRawMemory();

        for (size_t i = 0; i < rowNum; ++i)
        {
            for (size_t j = 0; j < colNum; ++j)
            {
                r[j] = (StorageType)(1 / (1 + exp(-r1[j])));
            }
            r1 += src1PackNum;
            r += tgtPackNum;
        }
        m_evalOutput.SetEval();
    }

private:
    TOperHandle m_oper;
    EvalHandle<Matrix<ElementType, DeviceType>> m_evalOutput;
};

struct Calculator
{
    template <typename TCaseTail, typename TEvalRes, typename TOperand>
    static void EvalRegister(TEvalRes& evalRes, const TOperand& oper)
    {
        static_assert(std::is_same<TCaseTail, OperSeqContainer<>>::value,
                      "General Case is not the last one");
                      
        using ElementType = typename TEvalRes::DataType::ElementType;
        using DeviceType = typename TEvalRes::DataType::DeviceType;

        auto handle = oper.EvalRegister();
        using UnitType = EvalUnit<decltype(handle), ElementType, DeviceType>;
        using GroupType = TrivalEvalGroup<UnitType>;

        auto outHandle = evalRes.Handle();
        const void* dataPtr = outHandle.DataPtr();
        UnitType unit(std::move(handle), std::move(outHandle));
        EvalPlan<DeviceType>::template Register<GroupType>(std::move(unit), dataPtr);
    }
};
}
}

template <>
struct OperBuildInSeq_<UnaryOpTags::Sigmoid>
{
    using type = OperSeqContainer<NSSigmoid::NSCaseGen::Calculator>;
};

template <typename TP>
struct OperSigmoid_
{
// valid check
private:
    using rawM = RemConstRef<TP>;

public:
    static constexpr bool valid = IsMatrix<rawM>;

public:
    static auto Eval(TP&& p_m)
    {
        using ResType = UnaryOp<UnaryOpTags::Sigmoid, rawM>;
        return ResType(std::forward<TP>(p_m));
    }
};

template <typename TP,
          std::enable_if_t<OperSigmoid_<TP>::valid>* = nullptr>
auto Sigmoid(TP&& p_m)
{
    return OperSigmoid_<TP>::Eval(std::forward<TP>(p_m));
}
}
