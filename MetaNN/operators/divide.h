#pragma once

#include <MetaNN/operators/facilities/tags.h>
#include <MetaNN/operators/operators.h>
#include <type_traits>
namespace MetaNN
{
namespace NSDivide
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

    EvalUnit(TOperHandle1&& oper1,
             TOperHandle2&& oper2,
             EvalHandle<Matrix<ElementType, DeviceType>> evalOutput)
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

        return (size_t)res;
    }
    
    void Eval() override
    {
        const auto& p_v1 = m_oper1.Data();
        const auto& p_v2 = m_oper2.Data();
        const size_t rowNum = p_v1.RowNum();
        const size_t colNum = p_v1.ColNum();
        assert(p_v2.RowNum() == rowNum);
        assert(p_v2.ColNum() == colNum);

        m_evalOutput.Allocate(rowNum, colNum);
        auto& res = m_evalOutput.MutableData();

        const auto mem_v1 = LowerAccess(p_v1);
        const auto mem_v2 = LowerAccess(p_v2);
        auto mem_res = LowerAccess(res);

        const size_t src1PackNum = mem_v1.RowLen();
        const size_t src2PackNum = mem_v2.RowLen();
        const size_t tgtPackNum = mem_res.RowLen();

        using StorageType = typename Scalar<ElementType, DeviceType>::StorageType;
        const StorageType* r1 = mem_v1.RawMemory();
        const StorageType* r2 = mem_v2.RawMemory();
        StorageType* r = mem_res.MutableRawMemory();

        for (size_t i = 0; i < rowNum; ++i)
        {
            for (size_t j = 0; j < colNum; ++j)
            {
                r[j] = r1[j] / r2[j];
            }
            r1 += src1PackNum;
            r2 += src2PackNum;
            r += tgtPackNum;
        }
        m_evalOutput.SetEval();
    }

private:
    TOperHandle1 m_oper1;
    TOperHandle2 m_oper2;
    EvalHandle<Matrix<ElementType, DeviceType>> m_evalOutput;
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
struct OperBuildInSeq_<BinaryOpTags::Divide>
{
    using type = OperSeqContainer<NSDivide::NSCaseGen::Calculator>;
};

template <typename TP1, typename TP2>
struct OperDivide_
{
// valid check
private:
    using rawM1 = RemConstRef<TP1>;
    using rawM2 = RemConstRef<TP2>;

public:
    static constexpr bool valid = (IsMatrix<rawM1> && IsMatrix<rawM2>) ||
                                  (IsMatrix<rawM1> && IsScalar<rawM2>) ||
                                  (IsScalar<rawM1> && IsMatrix<rawM2>);

public:
    template <typename T1, typename T2,
              std::enable_if_t<std::is_same<CategoryTags::Matrix, T1>::value>* = nullptr,
              std::enable_if_t<std::is_same<CategoryTags::Matrix, T2>::value>* = nullptr>
    static auto Eval(TP1&& p_m1, TP2&& p_m2)
    {
        static_assert(std::is_same<typename rawM1::ElementType, typename rawM2::ElementType>::value,
                      "Matrices with different element types cannot divide directly");
        static_assert(std::is_same<typename rawM1::DeviceType, typename rawM2::DeviceType>::value,
                      "Matrices with different device types cannot divide directly");

        using ResType = BinaryOp<BinaryOpTags::Divide, rawM1, rawM2>;
        return ResType(std::forward<TP1>(p_m1), std::forward<TP2>(p_m2));
    }

    template<typename T1, typename T2, typename TElem,
             std::enable_if_t<std::is_same<CategoryTags::Matrix, T1>::value>* = nullptr>
    static auto Eval(TP1&& p_m1, Scalar<TElem, DeviceTags::CPU>&& p_m2)
    {
        using ElementType = typename rawM1::ElementType;
        using DeviceType = typename rawM1::DeviceType;

        TrivalMatrix<ElementType, DeviceType> tmpMatrix(p_m1.RowNum(), p_m1.ColNum(),
                                                        p_m2.Value());
                                           
        using ResType = BinaryOp<BinaryOpTags::Divide,
                                 rawM1,
                                 TrivalMatrix<ElementType, DeviceType>>;
        return ResType(std::forward<TP1>(p_m1), std::move(tmpMatrix));
    }

    template<typename T1, typename T2, typename TElem,
             std::enable_if_t<std::is_same<CategoryTags::Matrix, T2>::value>* = nullptr>
    static auto Eval(Scalar<TElem, DeviceTags::CPU>&& p_m1, TP2&& p_m2)
    {
        using ElementType = typename rawM1::ElementType;
        using DeviceType = typename rawM1::DeviceType;

        TrivalMatrix<ElementType, DeviceType> tmpMatrix(p_m2.RowNum(), p_m2.ColNum(),
                                              p_m1.Value());

        using ResType = BinaryOp<BinaryOpTags::Divide,
                                 TrivalMatrix<ElementType, DeviceType>,
                                 rawM2>;
        return ResType(std::move(tmpMatrix), std::forward<TP2>(p_m2));
    }
};

template <typename TP1, typename TP2,
          std::enable_if_t<OperDivide_<TP1, TP2>::valid>* = nullptr>
auto operator/ (TP1&& p_m1, TP2&& p_m2)
{
    return OperDivide_<TP1, TP2>::
            template Eval<DataCategory<TP1>, DataCategory<TP2>>(std::forward<TP1>(p_m1), std::forward<TP2>(p_m2));
}
}
