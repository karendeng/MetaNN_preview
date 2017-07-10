#pragma once

#include <type_traits>
#include <MetaNN/operators/operators.h>
#include <cmath>

namespace MetaNN
{
namespace NSSigmoidDerivative
{
template <typename TOperHandle1, typename TOperHandle2, typename TDevice>
class EvalUnit;

template <typename TOperHandle1, typename TOperHandle2>
class EvalUnit<TOperHandle1, TOperHandle2, DeviceTags::CPU>
    : public BaseEvalUnit<DeviceTags::CPU>
{
    using TOperandData = std::decay_t<decltype(std::declval<TOperHandle1>().Data())>;
public:
    using ElementType = typename TOperandData::ElementType;
    using DeviceType = typename TOperandData::DeviceType;
    static_assert(std::is_same<DeviceType, DeviceTags::CPU>::value,
                  "Device type mismatch");

    EvalUnit(TOperHandle1 oper1,
             TOperHandle2 oper2,
             EvalHandle<Matrix<ElementType, DeviceType>> evalOutput)
        : m_oper1(std::move(oper1))
        , m_oper2(std::move(oper2))
        , m_evalOutput(std::move(evalOutput)) { }

    size_t OperandDepth(const std::unordered_map<const void*, size_t>& depMap) const
    {
        int res = -1;

        auto it = depMap.find(m_oper1.DataPtr());
        if (it != depMap.end()) res = std::max(res, (int)(it->second));

        it = depMap.find(m_oper1.DataPtr());
        if (it != depMap.end()) res = std::max(res, (int)(it->second));

        return (size_t)res;
    }
    
    void Eval() override
    {
        const auto& p_grad = m_oper1.Data();
        const auto& p_out = m_oper2.Data();

        const size_t rowNum = p_grad.RowNum();
        const size_t colNum = p_grad.ColNum();
        assert(p_out.RowNum() == rowNum);
        assert(p_out.ColNum() == colNum);
        
        m_evalOutput.Allocate(rowNum, colNum);
        auto& res = m_evalOutput.Data();

        auto mem_grad = LowerAccess(p_grad);
        auto mem_out = LowerAccess(p_out);
        auto mem_res = LowerAccess(res);

        const size_t srcGradPackNum = mem_grad.RowLen();
        const size_t srcOutPackNum = mem_out.RowLen();
        const size_t tgtPackNum = mem_res.RowLen();

        const ElementType* r1 = mem_grad.RawMemory();
        const ElementType* r2 = mem_out.RawMemory();
        ElementType* r = mem_res.MutableRawMemory();

        for (size_t i = 0; i < rowNum; ++i)
        {
            for (size_t j = 0; j < colNum; ++j)
            {
                r[j] = r1[j] * r2[j] * (1 - r2[j]);
            }
            r1 += srcGradPackNum;
            r2 += srcOutPackNum;
            r += tgtPackNum;
        }
    }

private:
    TOperHandle1 m_oper1;
    TOperHandle2 m_oper2;
    EvalHandle<Matrix<ElementType, DeviceType>> m_evalOutput;
};

struct GeneralCase
{
    template <typename TCaseTail, typename TEvalRes, typename TOperator1, typename TOperator2>
    static void EvalRegister(TEvalRes& evalRes, const TOperator1& oper1, const TOperator2& oper2)
    {
        static_assert(std::is_same<TCaseTail, OperSeqContainer<>>::value,
                      "General Case is not the last one");
                      
        using RawEvalRes = typename TEvalRes::DataType;
        using DeviceType = typename RawEvalRes::DeviceType;

        auto handle1 = oper1.EvalRegister();
        auto handle2 = oper2.EvalRegister();
        using UnitType = EvalUnit<decltype(handle1), decltype(handle2), DeviceType>;
        using GroupType = TrivalEvalGroup<UnitType>;

        auto outHandle = evalRes.Handle();
        const void* dataPtr = outHandle.DataPtr();
        UnitType unit(std::move(handle1), std::move(handle2), std::move(outHandle));
        EvalPlan<DeviceType>::template Register<GroupType>(std::move(unit), dataPtr);
    }
};
}

template <>
struct OperBuildInSeq_<BinaryOpTags::SigmoidDerivative>
{
    using type = OperSeqContainer<NSSigmoidDerivative::GeneralCase>;
};

template <typename TGrad, typename TOut>
struct OperSigmoidDerivative_
{
private:
    using rawM1 = std::decay_t<TGrad>;
    using rawM2 = std::decay_t<TOut>;

public:
    static constexpr bool valid = (IsMatrix<rawM1> && IsMatrix<rawM2>);

public:
    static auto Eval(TGrad&& p_grad, TOut&& p_out)
    {
        using ResType = BinaryOp<BinaryOpTags::SigmoidDerivative, rawM1, rawM2>;
        return ResType(std::forward<TGrad>(p_grad), std::forward<TOut>(p_out));
    }
};

template <typename TGrad, typename TOut,
          std::enable_if_t<OperSigmoidDerivative_<TGrad, TOut>::valid>* = nullptr>
auto SigmoidDerivative(TGrad&& p_grad, TOut&& p_out)
{
    return OperSigmoidDerivative_<TGrad, TOut>::Eval(std::forward<TGrad>(p_grad),
                                                      std::forward<TOut>(p_out));
}
}
