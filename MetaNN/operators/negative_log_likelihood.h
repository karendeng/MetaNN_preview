#pragma once
#include <type_traits>
#include <vector>
#include <cmath>

namespace MetaNN
{
template <>
struct OperCategory_<BinaryOpTags::NegativeLogLikelihood,
                     CategoryTags::Matrix,
                     CategoryTags::Matrix>
{
    using type = CategoryTags::Scalar;
};

namespace NSNegativeLogLikelihood
{
namespace NSCaseGen
{
template <typename TOperHandle1, typename TOperHandle2, typename TElem, typename TDevice>
class EvalUnit;

template <typename TOperHandle1, typename TOperHandle2, typename TElem>
class EvalUnit<TOperHandle1, TOperHandle2, TElem, DeviceTags::CPU>
    : public BaseEvalUnit<DeviceTags::CPU>
{
public:
    using ElementType = TElem;
    using DeviceType = DeviceTags::CPU;

    EvalUnit(TOperHandle1 oper1,
             TOperHandle2 oper2,
             EvalHandle<Scalar<ElementType, DeviceType>> evalOutput)
        : m_oper1(std::move(oper1))
        , m_oper2(std::move(oper2))
        , m_evalOutput(evalOutput) { }

    size_t OperandDepth(const std::unordered_map<const void*, size_t>& depMap) const
    {
        int res = -1;

        auto it = depMap.find(m_oper1.DataPtr());
        if (it != depMap.end()) res = std::max(res, (int)(it->second));

        it = depMap.find(m_oper2.DataPtr());
        if (it != depMap.end()) res = std::max(res, (int)(it->second));

        return res;
    }
    
    void Eval() override
    {
        const auto& p_tar = m_oper1.Data();
        const auto& p_pre = m_oper2.Data();
        m_evalOutput.Allocate();

        const size_t rowNum = p_tar.RowNum();
        const size_t colNum = p_tar.ColNum();
        assert(p_pre.RowNum() == rowNum);
        assert(p_pre.ColNum() == colNum);

        using StorageType = typename Scalar<ElementType, DeviceType>::StorageType;
        auto res = StorageType();

        auto mem_v1 = LowerAccess(p_tar);
        auto mem_v2 = LowerAccess(p_pre);

        const size_t src1PackNum = mem_v1.RowLen();
        const size_t src2PackNum = mem_v2.RowLen();

        const StorageType* r1 = mem_v1.RawMemory();
        const StorageType* r2 = mem_v2.RawMemory();

        for (size_t i = 0; i < rowNum; ++i)
        {
            for (size_t j = 0; j < colNum; ++j)
            {
                res -= r1[j] * log(r2[j]);
            }
            r1 += src1PackNum;
            r2 += src2PackNum;
        }
        
        m_evalOutput.MutableData().Value() = res;
        m_evalOutput.SetEval();
    }

private:
    TOperHandle1 m_oper1;
    TOperHandle2 m_oper2;
    EvalHandle<Scalar<ElementType, DeviceType>> m_evalOutput;
};

struct Calculator
{
    template <typename TCaseTail, typename TEvalRes, typename TOperator1, typename TOperator2>
    static void EvalRegister(TEvalRes& evalRes, const TOperator1& oper1, const TOperator2& oper2)
    {
        static_assert(std::is_same<TCaseTail, OperSeqContainer<>>::value,
                      "General Case is not the last one");
                      
        using ElementType = typename TEvalRes::DataType::ElementType;
        using DeviceType = typename TEvalRes::DataType::DeviceType;

        auto handle1 = oper1.EvalRegister();
        auto handle2 = oper2.EvalRegister();
        using UnitType = EvalUnit<decltype(handle1), decltype(handle2), ElementType, DeviceType>;
        using GroupType = TrivalEvalGroup<UnitType>;

        auto outHandle = evalRes.Handle();
        const void* dataPtr = outHandle.DataPtr();
        UnitType unit(std::move(handle1), std::move(handle2), std::move(outHandle));
        EvalPlan<DeviceType>::template Register<GroupType>(std::move(unit), dataPtr);
    }
};
}
}

template <>
struct OperBuildInSeq_<BinaryOpTags::NegativeLogLikelihood>
{
    using type = OperSeqContainer<NSNegativeLogLikelihood::NSCaseGen::Calculator>;
};

template <typename TP1, typename TP2>
struct OperNegativeLogLikelihood_
{
// valid check
private:
    using rawM1 = RemConstRef<TP1>;
    using rawM2 = RemConstRef<TP2>;

public:
    static constexpr bool valid = (IsMatrix<rawM1> && IsMatrix<rawM2>);

public:
    static auto Eval(TP1&& p_m1, TP2&& p_m2)
    {
        static_assert(std::is_same<typename rawM1::ElementType, typename rawM2::ElementType>::value,
                      "Matrices with different element types cannot do NegativeLogLikelihood directly");
        static_assert(std::is_same<typename rawM1::DeviceType, typename rawM2::DeviceType>::value,
                      "Matrices with different device types cannot do NegativeLogLikelihood directly");

        using ResType = BinaryOp<BinaryOpTags::NegativeLogLikelihood, rawM1, rawM2>;
        return ResType(std::forward<TP1>(p_m1), std::forward<TP2>(p_m2));
    }
};

template <typename TP1, typename TP2,
          std::enable_if_t<OperNegativeLogLikelihood_<TP1, TP2>::valid>* = nullptr>
auto NegativeLogLikelihood(TP1&& p_tar, TP2&& p_pre)
{
    return OperNegativeLogLikelihood_<TP1, TP2>::Eval(std::forward<TP1>(p_tar), std::forward<TP2>(p_pre));
}
}
