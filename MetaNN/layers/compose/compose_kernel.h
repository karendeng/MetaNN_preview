#pragma once
#include <MetaNN/policies/change_policy.h>

namespace MetaNN
{
template <typename TLayerTag, template<typename> class TLayer>
struct SubLayer
{
    using Tag = TLayerTag;

    template <typename T> using Layer = TLayer<T>;
};

template <typename TOutLayerTag, typename TOutName, typename TInLayerTag, typename TInName>
struct InternalConnect
{
    using OutTag = TOutLayerTag;
    using OutName = TOutName;
    using InTag = TInLayerTag;
    using InName = TInName;
};

template <typename TInName, typename TInLayerTag, typename TInLayerName>
struct InConnect
{
    using InName = TInName;
    using InLayerTag = TInLayerTag;
    using InLayerName = TInLayerName;
};

template <typename TOutLayerTag, typename TOutLayerName, typename TOutName>
struct OutConnect
{
    using OutLayerTag = TOutLayerTag;
    using OutLayerName = TOutLayerName;
    using OutName = TOutName;
};

namespace NSComposeKernel
{
template <typename TLayerTag, typename TLayer>
struct InstantiatedSublayer
{
    using Tag = TLayerTag;
    using Layer = TLayer;
};

template <typename...T> struct SubLayerContainer;
template <typename...T> struct InterConnectContainer;
template <typename...T> struct InConnectContainer;
template <typename...T> struct OutConnectContainer;
template <typename...T> struct InstantiatedSublayerContainer;

template <typename TLayerTag, template<typename> class TLayer, typename TPC>
struct SublayerPolicies
{
    using Tag = TLayerTag;
    template <typename T> using Layer = TLayer<T>;
    using Policy = TPC;
};

template <typename...> struct SublayerPolicyContainer;

/// ======================== Separate parameters ========================
template <typename...TParameters>
struct SeparateParametersHelper
{
    template<typename TS, typename TIt, typename TI, typename TO, typename...T>
    struct SeparateHelper
    {
        static_assert(sizeof...(T) == 0, "Error Compose Parameter");

        using SubLayerRes = TS;
        using InterConnectRes = TIt;
        using InConnectRes = TI;
        using OutConnectRes = TO;
    };

    template <typename...TS, typename...TIt, typename...TI, typename...TO,
              typename TLayerTag, template<typename> class TLayer, typename...T>
    struct SeparateHelper<SubLayerContainer<TS...>, InterConnectContainer<TIt...>,
                          InConnectContainer<TI...>, OutConnectContainer<TO...>,
                          SubLayer<TLayerTag, TLayer>, T...>
    {
        using Res = SeparateHelper<SubLayerContainer<TS..., SubLayer<TLayerTag, TLayer>>,
                                   InterConnectContainer<TIt...>, InConnectContainer<TI...>,
                                   OutConnectContainer<TO...>, T...>;

        using SubLayerRes = typename Res::SubLayerRes;
        using InterConnectRes = typename Res::InterConnectRes;
        using InConnectRes = typename Res::InConnectRes;
        using OutConnectRes = typename Res::OutConnectRes;
    };

    template <typename...TS, typename...TIt, typename...TI, typename...TO,
              typename TOutLayerTag, typename TOutName, typename TInLayerTag, typename TInName,
              typename...T>
    struct SeparateHelper<SubLayerContainer<TS...>, InterConnectContainer<TIt...>,
                          InConnectContainer<TI...>, OutConnectContainer<TO...>,
                          InternalConnect<TOutLayerTag, TOutName, TInLayerTag, TInName>, T...>
    {
        using Res = SeparateHelper<SubLayerContainer<TS...>,
                                   InterConnectContainer<TIt..., InternalConnect<TOutLayerTag, TOutName, TInLayerTag, TInName>>,
                                   InConnectContainer<TI...>,
                                   OutConnectContainer<TO...>, T...>;

        using SubLayerRes = typename Res::SubLayerRes;
        using InterConnectRes = typename Res::InterConnectRes;
        using InConnectRes = typename Res::InConnectRes;
        using OutConnectRes = typename Res::OutConnectRes;
    };

    template <typename...TS, typename...TIt, typename...TI, typename...TO,
              typename TInName, typename TInLayerTag, typename TInLayerName,
              typename...T>
    struct SeparateHelper<SubLayerContainer<TS...>, InterConnectContainer<TIt...>,
                          InConnectContainer<TI...>, OutConnectContainer<TO...>,
                          InConnect<TInName, TInLayerTag, TInLayerName>, T...>
    {
        using Res = SeparateHelper<SubLayerContainer<TS...>,
                                   InterConnectContainer<TIt...>,
                                   InConnectContainer<TI..., InConnect<TInName, TInLayerTag, TInLayerName>>,
                                   OutConnectContainer<TO...>, T...>;

        using SubLayerRes = typename Res::SubLayerRes;
        using InterConnectRes = typename Res::InterConnectRes;
        using InConnectRes = typename Res::InConnectRes;
        using OutConnectRes = typename Res::OutConnectRes;
    };

    template <typename...TS, typename...TIt, typename...TI, typename...TO,
              typename TOutLayerTag, typename TOutLayerName, typename TOutName,
              typename...T>
    struct SeparateHelper<SubLayerContainer<TS...>, InterConnectContainer<TIt...>,
                          InConnectContainer<TI...>, OutConnectContainer<TO...>,
                          OutConnect<TOutLayerTag, TOutLayerName, TOutName>, T...>
    {
        using Res = SeparateHelper<SubLayerContainer<TS...>,
                                   InterConnectContainer<TIt...>,
                                   InConnectContainer<TI...>,
                                   OutConnectContainer<TO..., OutConnect<TOutLayerTag, TOutLayerName, TOutName>>,
                                   T...>;

        using SubLayerRes = typename Res::SubLayerRes;
        using InterConnectRes = typename Res::InterConnectRes;
        using InConnectRes = typename Res::InConnectRes;
        using OutConnectRes = typename Res::OutConnectRes;
    };

    using tmp = SeparateHelper<SubLayerContainer<>, InterConnectContainer<>,
                               InConnectContainer<>, OutConnectContainer<>, TParameters...>;

    using SubLayerRes = typename tmp::SubLayerRes;
    using InterConnectRes = typename tmp::InterConnectRes;
    using InConnectRes = typename tmp::InConnectRes;
    using OutConnectRes = typename tmp::OutConnectRes;
};

template <typename...TComposeKernelInfo>
using SeparateParameters = SeparateParametersHelper<TComposeKernelInfo...>;

/// =========== Check tag exist in arrays ==============================
template <typename TCheckTag, typename...TArray>
struct TagExist
{
    template <typename...T>
    struct imp
    {
        constexpr static bool value = false;
    };

    template <typename TCur, typename...T>
    struct imp<TCur, T...>
    {
        constexpr static bool tmp = std::is_same<TCheckTag, TCur>::value;
        constexpr static bool value = OrValue<tmp, imp<T...>>;
    };
    constexpr static bool value = imp<TArray...>::value;
};

template <typename TCheckTag, typename...TArray>
struct TagExistInLayerComps
{
    template <typename...T>
    struct imp
    {
        constexpr static bool value = false;
    };

    template <typename TLayerTag, template<typename> class TLayer, typename...T>
    struct imp<SubLayer<TLayerTag, TLayer>, T...>
    {
        constexpr static bool tmp = std::is_same<TCheckTag, TLayerTag>::value;
        constexpr static bool value = OrValue<tmp, imp<T...>>;
    };

    template <typename TOutLayerTag, typename TOutName,
              typename TInLayerTag, typename TInName, typename...T>
    struct imp<InternalConnect<TOutLayerTag, TOutName, TInLayerTag, TInName>, T...>
    {
        constexpr static bool tmp1 = std::is_same<TCheckTag, TOutLayerTag>::value;
        constexpr static bool tmp2 = std::is_same<TCheckTag, TInLayerTag>::value;
        constexpr static bool tmp = tmp1 || tmp2;
        constexpr static bool value = OrValue<tmp, imp<T...>>;
    };

    template <typename TInName, typename TInLayerTag, typename TInLayerName, typename...T>
    struct imp<InConnect<TInName, TInLayerTag, TInLayerName>, T...>
    {
        constexpr static bool tmp = std::is_same<TCheckTag, TInLayerTag>::value;
        constexpr static bool value = OrValue<tmp, imp<T...>>;
    };

    template <typename TOutLayerTag, typename TOutLayerName, typename TOutName, typename...T>
    struct imp<OutConnect<TOutLayerTag, TOutLayerName, TOutName>, T...>
    {
        constexpr static bool tmp = std::is_same<TCheckTag, TOutLayerTag>::value;
        constexpr static bool value = OrValue<tmp, imp<T...>>;
    };

    constexpr static bool value = imp<TArray...>::value;
};

/// ========= Sublayer check ===================================================
template <typename TSublayerCont> struct SublayerCheck;
template <typename...TSublayers>
struct SublayerCheck<SubLayerContainer<TSublayers...>>
{
    template <typename...T>
    struct CheckUniqueLayerTag
    {
        static constexpr bool value = true;
    };

    template <typename TSubLayer, typename...T>
    struct CheckUniqueLayerTag<TSubLayer, T...>
    {
        using CurTag = typename TSubLayer::Tag;

        static constexpr bool tmp = !(TagExistInLayerComps<CurTag, T...>::value);
        static constexpr bool value = AndValue<tmp, CheckUniqueLayerTag<T...>>;
    };

    constexpr static bool IsUnique = CheckUniqueLayerTag<TSublayers...>::value;
};

/// ========= Internal Connection Check ===============================================
template <typename TInternalCont> struct InternalConnectCheck;

template <typename...TInternalConnects>
struct InternalConnectCheck<InterConnectContainer<TInternalConnects...>>
{
    template <typename TInTag, typename TInName, typename...T>
    struct UniqueConnectCheck
    {
        constexpr static bool value = true;
    };

    template <typename TInTag, typename TInName, typename TCur, typename...T>
    struct UniqueConnectCheck<TInTag, TInName, TCur, T...>
    {
        using CheckTag = typename TCur::InTag;
        using CheckName = typename TCur::InName;

        static constexpr bool tmp1 = !(std::is_same<TInTag, CheckTag>::value);
        static constexpr bool tmp2 = !(std::is_same<TInName, CheckName>::value);
        static constexpr bool tmp3 = tmp1 || tmp2;
        static constexpr bool value = AndValue<tmp3, UniqueConnectCheck<TInTag, TInName, T...>>;
    };

    template <typename...T>
    struct UniqueSourceCheck
    {
        static constexpr bool value = true;
    };

    template <typename TCur, typename...T>
    struct UniqueSourceCheck<TCur, T...>
    {
        using CheckTag = typename TCur::InTag;
        using CheckName = typename TCur::InName;

        static constexpr bool tmp1 = UniqueConnectCheck<CheckTag, CheckName, T...>::value;
        static constexpr bool value = AndValue<tmp1, UniqueSourceCheck<T...>>;
    };

    template <typename...T>
    struct NoCycleCheck
    {
        static constexpr bool value = true;
    };

    template <typename TCur, typename...T>
    struct NoCycleCheck<TCur, T...>
    {
        using CheckIn = typename TCur::InTag;
        using CheckOut = typename TCur::OutTag;
        static constexpr bool tmp = !(std::is_same<CheckIn, CheckOut>::value);
        static constexpr bool value = AndValue<tmp, NoCycleCheck<T...>>;
    };

    constexpr static bool NoSelfCycle = NoCycleCheck<TInternalConnects...>::value;
    constexpr static bool UniqueSource = UniqueSourceCheck<TInternalConnects...>::value;
};

/// ========= Input Connection Check ===================================================
template <typename TInputCont> struct InputConnectCheck;

template <typename...TInputConnects>
struct InputConnectCheck<InConnectContainer<TInputConnects...>>
{
    template <typename TInLayerTag, typename TInLayerName, typename...T>
    struct UniqueConnectCheckHelper
    {
        constexpr static bool value = true;
    };

    template <typename TInLayerTag, typename TInLayerName, typename TCur, typename...T>
    struct UniqueConnectCheckHelper<TInLayerTag, TInLayerName, TCur, T...>
    {
        using CheckTag = typename TCur::InLayerTag;
        using CheckName = typename TCur::InLayerName;

        static constexpr bool tmp1 = !(std::is_same<TInLayerTag, CheckTag>::value);
        static constexpr bool tmp2 = !(std::is_same<TInLayerName, CheckName>::value);
        static constexpr bool tmp3 = tmp1 || tmp2;
        static constexpr bool value
            = AndValue<tmp3, UniqueConnectCheckHelper<TInLayerTag, TInLayerName, T...>>;
    };

    template <typename...T>
    struct UniqueConnectCheck
    {
        static constexpr bool value = true;
    };

    template <typename TCur, typename...T>
    struct UniqueConnectCheck<TCur, T...>
    {
        using CheckTag = typename TCur::InLayerTag;
        using CheckName = typename TCur::InLayerName;

        static constexpr bool tmp = UniqueConnectCheckHelper<CheckTag, CheckName, T...>::value;
        static constexpr bool value = AndValue<tmp, UniqueConnectCheck<T...>>;
    };
    constexpr static bool UniqueSource = UniqueConnectCheck<TInputConnects...>::value;
};

/// ========= Output Connection Check ==================================================
template <typename TOutputCont> struct OutputConnectCheck;

template <typename...TOutputConnects>
struct OutputConnectCheck<OutConnectContainer<TOutputConnects...>>
{
    template <typename TOutName, typename...T>
    struct UniqueOutputNameCheckHelper
    {
        constexpr static bool value = true;
    };

    template <typename TOutName, typename TCur, typename...T>
    struct UniqueOutputNameCheckHelper<TOutName, TCur, T...>
    {
        using CheckName = typename TCur::OutName;
        static constexpr bool tmp = !(std::is_same<TOutName, CheckName>::value);
        static constexpr bool value = AndValue<tmp, UniqueOutputNameCheckHelper<TOutName, T...>>;
    };

    template <typename...T>
    struct UniqueOutputNameCheck
    {
        static constexpr bool value = true;
    };

    template <typename TCur, typename...T>
    struct UniqueOutputNameCheck<TCur, T...>
    {
        using OutputName = typename TCur::OutName;
        static constexpr bool tmp = UniqueOutputNameCheckHelper<OutputName, T...>::value;
        static constexpr bool value = AndValue<tmp, UniqueOutputNameCheck<T...>>;
    };
    constexpr static bool UniqueSource = UniqueOutputNameCheck<TOutputConnects...>::value;
};

/// ========= Internal Tag Should In Sublayer ==========================================
template <typename TInternalArray, typename TSubLayerArray> struct InternalTagInSublayer;

template <typename...TInterTags, typename...TSubLayers>
struct InternalTagInSublayer<InterConnectContainer<TInterTags...>, SubLayerContainer<TSubLayers...>>
{
    template <typename...T>
    struct imp
    {
        static constexpr bool value = true;
    };

    template <typename TInterElement, typename...T>
    struct imp<TInterElement, T...>
    {
        using CurIn = typename TInterElement::InTag;
        using CurOut = typename TInterElement::OutTag;
        static constexpr bool tmp1 = TagExistInLayerComps<CurOut, TSubLayers...>::value;
        static constexpr bool tmp2 = TagExistInLayerComps<CurIn, TSubLayers...>::value;
        static constexpr bool tmp3 = tmp1 && tmp2;
        static constexpr bool value = AndValue<tmp3, imp<T...>>;
    };

    constexpr static bool value = imp<TInterTags...>::value;
};

/// ========= Input Tag Should In Sublayer =============================================
template <typename TInputArray, typename TSublayerArray> struct InputTagInSubLayer;

template <typename...TInputTags, typename...TSubLayers>
struct InputTagInSubLayer<InConnectContainer<TInputTags...>, SubLayerContainer<TSubLayers...>>
{
    template <typename...T>
    struct imp
    {
        static constexpr bool value = true;
    };

    template <typename TInputElement, typename...T>
    struct imp<TInputElement, T...>
    {
        using CurIn = typename TInputElement::InLayerTag;
        static constexpr bool tmp = TagExistInLayerComps<CurIn, TSubLayers...>::value;
        static constexpr bool value = AndValue<tmp, imp<T...>>;
    };
    constexpr static bool value = imp<TInputTags...>::value;
};

/// ========= Output Tag Should In Sublayer ============================================
template <typename TOutputArray, typename TSublayerArray> struct OutputTagInSubLayer;

template <typename...TOutputTags, typename...TSubLayers>
struct OutputTagInSubLayer<OutConnectContainer<TOutputTags...>, SubLayerContainer<TSubLayers...>>
{
    template <typename...T>
    struct imp
    {
        static constexpr bool value = true;
    };

    template <typename TOutputElement, typename...T>
    struct imp<TOutputElement, T...>
    {
        using CurOut = typename TOutputElement::OutLayerTag;
        static constexpr bool tmp = TagExistInLayerComps<CurOut, TSubLayers...>::value;
        static constexpr bool value = AndValue<tmp, imp<T...>>;
    };
    constexpr static bool value = imp<TOutputTags...>::value;
};

/// ========= Sublayers' Tags Sould Exist in Other Arrays ==============================
template <typename TInterArray, typename TInArray, typename TOutArray,
          typename TSublayerArray> struct SublayerTagInOtherArrays;

template <typename...TInterElems, typename...TInElems, typename...TOutElems,
          typename...TSublayerElems>
struct SublayerTagInOtherArrays<InterConnectContainer<TInterElems...>,
                                InConnectContainer<TInElems...>,
                                OutConnectContainer<TOutElems...>,
                                SubLayerContainer<TSublayerElems...>>
{
    template <typename...T>
    struct imp
    {
        static constexpr bool value = true;
    };

    template <typename TCur, typename...T>
    struct imp<TCur, T...>
    {
        using CurLayerTag = typename TCur::Tag;
        static constexpr bool tmp1 = TagExistInLayerComps<CurLayerTag, TInterElems...>::value;
        static constexpr bool tmp2 = TagExistInLayerComps<CurLayerTag, TInElems...>::value;
        static constexpr bool tmp3 = TagExistInLayerComps<CurLayerTag, TOutElems...>::value;
        static constexpr bool tmp4 = tmp1 || tmp2 || tmp3;
        static constexpr bool value = AndValue<tmp4, imp<T...>>;
    };

    constexpr static bool value = imp<TSublayerElems...>::value;
};

/// ========= Whether Tag In Internal Connections' Out-Tag or In-Tag =====================
template <typename TCheckTag, typename...TInternalConnects>
struct TagInInternalOut
{
    template <typename...T>
    struct imp
    {
        constexpr static bool value = false;
    };

    template <typename TCur, typename...T>
    struct imp<TCur, T...>
    {
        using CurTag = typename TCur::OutTag;
        constexpr static bool tmp = std::is_same<TCheckTag, CurTag>::value;
        constexpr static bool value = OrValue<tmp, imp<T...>>;
    };

    constexpr static bool value = imp<TInternalConnects...>::value;
};

template <typename TCheckTag, typename...TInternalConnects>
struct TagInInternalIn
{
    template <typename...T>
    struct imp
    {
        constexpr static bool value = false;
    };

    template <typename TCur, typename...T>
    struct imp<TCur, T...>
    {
        using CurTag = typename TCur::InTag;
        constexpr static bool tmp = std::is_same<TCheckTag, CurTag>::value;
        constexpr static bool value = OrValue<tmp, imp<T...>>;
    };

    constexpr static bool value = imp<TInternalConnects...>::value;
};

/// ========= Internal Post Layers Should Be Useful ==================================
template <typename TInterArray, typename TOutArray> struct UsefulInternalPostLayer;

template <typename...TInterElems, typename...TOutElems>
struct UsefulInternalPostLayer<InterConnectContainer<TInterElems...>,
                               OutConnectContainer<TOutElems...>>
{
    template <typename...T>
    struct imp
    {
        static constexpr bool value = true;
    };

    template <typename TCur, typename...T>
    struct imp<TCur, T...>
    {
        using CurCheckTag = typename TCur::InTag;
        static constexpr bool tmp1 = TagInInternalOut<CurCheckTag, TInterElems...>::value;
        static constexpr bool tmp2 = TagExistInLayerComps<CurCheckTag, TOutElems...>::value;
        static constexpr bool tmp3 = tmp1 || tmp2;
        static constexpr bool value = AndValue<tmp3, imp<T...>>;
    };
    constexpr static bool value = imp<TInterElems...>::value;
};

/// ========= Input Layers Should Be Useful ============================================
template <typename TInArray, typename TInterArray, typename TOutArray> struct UsefulInputLayer;

template <typename...TInElems, typename...TInterElems, typename...TOutElems>
struct UsefulInputLayer<InConnectContainer<TInElems...>,
                        InterConnectContainer<TInterElems...>,
                        OutConnectContainer<TOutElems...>>
{
    template <typename...T>
    struct imp
    {
        static constexpr bool value = true;
    };

    template <typename TCur, typename...T>
    struct imp<TCur, T...>
    {
        using CurCheckTag = typename TCur::InLayerTag;
        static constexpr bool tmp1 = TagInInternalOut<CurCheckTag, TInterElems...>::value;
        static constexpr bool tmp2 = TagExistInLayerComps<CurCheckTag, TOutElems...>::value;
        static constexpr bool tmp3 = tmp1 || tmp2;
        static constexpr bool value = AndValue<tmp3, imp<T...>>;
    };
    constexpr static bool value = imp<TInElems...>::value;
};

/// ========= Topological Ordering Implementation ======================================
namespace NSTPO
{
template <typename...T> struct TagContainer;

template <typename TRemainInters, typename TCheckInters, typename TPostTags, typename...T>
struct InternalLayerPrune
{
    using PostTags = TPostTags;
    using type = TRemainInters;
};

template <typename...T1, typename...T2, typename...TTags, typename TCur, typename...T3>
struct InternalLayerPrune<InterConnectContainer<T1...>,
                          InterConnectContainer<T2...>,
                          TagContainer<TTags...>,
                          TCur, T3...>
{
    using CheckTag = typename TCur::OutTag;
    static constexpr bool inInterIn = TagInInternalIn<CheckTag, T2...>::value;

    template <bool isCheckOk, typename TDummy = void>
    struct put
    {
        using NewTagContainer = TagContainer<TTags...>;
        using type = InterConnectContainer<T1..., TCur>;
    };

    template <typename TDummy>
    struct put<false, TDummy>
    {
        constexpr static bool CheckTagInTTags = TagExist<CheckTag, TTags...>::value;
        using NewTagContainer = std::conditional_t<CheckTagInTTags,
                                                   TagContainer<TTags...>,
                                                   TagContainer<TTags..., CheckTag>>;
        using type = InterConnectContainer<T1...>;
    };

    using tmp = typename put<inInterIn>::type;
    using tmpTagContainer = typename put<inInterIn>::NewTagContainer;

    using nextStep = InternalLayerPrune<tmp, InterConnectContainer<T2...>, tmpTagContainer, T3...>;

    using type = typename nextStep::type;
    using PostTags = typename nextStep::PostTags;
};

template <typename TInputSubLayers,
          typename TOrderedSublayers, typename TUnorderedSublayers,
          typename TPostTags>
struct SeparateByPostTag
{
    using Ordered = TOrderedSublayers;
    using Unordered = TUnorderedSublayers;
};

template <typename TCur, typename...TSI, typename...TSO, typename...TSN, typename...TPostTagElems>
struct SeparateByPostTag<SubLayerContainer<TCur, TSI...>,
                         SubLayerContainer<TSO...>, SubLayerContainer<TSN...>,
                         TagContainer<TPostTagElems...>>
{
    using CheckTag = typename TCur::Tag;
    constexpr static bool IsExist = TagExist<CheckTag, TPostTagElems...>::value;

    using tmpOrdered = std::conditional_t<IsExist,
                                          SubLayerContainer<TSO..., TCur>,
                                          SubLayerContainer<TSO...>>;
    using tmpUnordered = std::conditional_t<IsExist,
                                            SubLayerContainer<TSN...>,
                                            SubLayerContainer<TSN..., TCur>>;

    using NextStep = SeparateByPostTag<SubLayerContainer<TSI...>, tmpOrdered,
                                       tmpUnordered, TagContainer<TPostTagElems...>>;
    using Ordered = typename NextStep::Ordered;
    using Unordered = typename NextStep::Unordered;
};

template <typename THeadSublayers, typename TTailSublayers>
struct CatSublayers
{
    using type = THeadSublayers;
};

template <typename...T1, typename TCur, typename...T2>
struct CatSublayers<SubLayerContainer<T1...>, SubLayerContainer<TCur, T2...>>
{
    using tmp = SubLayerContainer<T1..., TCur>;
    using type = typename CatSublayers<tmp, SubLayerContainer<T2...>>::type;
};

template <typename TOrderedSublayers, typename TUnorderedSublayers,
          typename TCheckInternals>
struct MainLoop
{
    static_assert((ArraySize<TCheckInternals> == 0), "Cycle exist in the compose layer");
    using Ordered = TOrderedSublayers;
    using Remain = TUnorderedSublayers;
};

template <typename...TSO, typename...TSN, typename TIC, typename...TI>
struct MainLoop<SubLayerContainer<TSO...>,
                SubLayerContainer<TSN...>,
                InterConnectContainer<TIC, TI...>>
{
    using CurInter = InterConnectContainer<TIC, TI...>;

    using NewInter = typename InternalLayerPrune<InterConnectContainer<>,
                                                 CurInter, TagContainer<>, TIC, TI...>::type;
    using PostTags = typename InternalLayerPrune<InterConnectContainer<>,
                                                 CurInter, TagContainer<>, TIC, TI...>::PostTags;
    static_assert((ArraySize<NewInter> < ArraySize<CurInter>),
                                                "Cycle exist in the compose layer");

    using SeparateByTagFun = SeparateByPostTag<SubLayerContainer<TSN...>,
                                               SubLayerContainer<TSO...>,
                                               SubLayerContainer<>, PostTags>;

    using NewOrdered = typename SeparateByTagFun::Ordered;
    using NewUnordered = typename SeparateByTagFun::Unordered;
    using Ordered = typename MainLoop<NewOrdered, NewUnordered, NewInter>::Ordered;
    using Remain = typename MainLoop<NewOrdered, NewUnordered, NewInter>::Remain;
};
}// namespace NSTPO
template <typename TSubLayerArray, typename TInterArray>
struct TopologicalOrdering_;

template <typename...TSubLayerElems, typename...TInterElems>
struct TopologicalOrdering_<SubLayerContainer<TSubLayerElems...>,
                           InterConnectContainer<TInterElems...>>
{
    template <typename TSublayerOrdered, typename TSubLayerUnordered, typename...T>
    struct SublayerPreprocess
    {
        using Ordered = TSublayerOrdered;
        using Unordered = TSubLayerUnordered;
    };

    template <typename...TSO, typename...TSN, typename TCur, typename...T>
    struct SublayerPreprocess<SubLayerContainer<TSO...>,
                              SubLayerContainer<TSN...>,
                              TCur, T...>
    {
        using CurTag = typename TCur::Tag;
        static constexpr bool inInter = TagExistInLayerComps<CurTag, TInterElems...>::value;

        using NewOrdered = std::conditional_t<inInter,
                                              SubLayerContainer<TSO...>,
                                              SubLayerContainer<TSO..., TCur>>;
        using NewUnordered = std::conditional_t<inInter,
                                                SubLayerContainer<TSN..., TCur>,
                                                SubLayerContainer<TSN...>>;

        using Ordered = typename SublayerPreprocess<NewOrdered, NewUnordered, T...>::Ordered;
        using Unordered = typename SublayerPreprocess<NewOrdered, NewUnordered, T...>::Unordered;
    };
    using SublayerPreprocessFun =
        SublayerPreprocess<SubLayerContainer<>, SubLayerContainer<>, TSubLayerElems...>;

    using OrderedAfterPreprocess = typename SublayerPreprocessFun::Ordered;
    using UnorderedAfterPreprocess = typename SublayerPreprocessFun::Unordered;

    using MainLoopFun = NSTPO::MainLoop<OrderedAfterPreprocess, UnorderedAfterPreprocess,
                                        InterConnectContainer<TInterElems...>>;

    using OrderedAfterMain = typename MainLoopFun::Ordered;
    using RemainAfterMain = typename MainLoopFun::Remain;

    using type = typename NSTPO::CatSublayers<OrderedAfterMain, RemainAfterMain>::type;
};

/// ========= Sublayer Instantiation ===================================================
namespace NSSI
{
template <typename TPolicyCont, typename TOrderedSublayerCont> struct GetSublayerPolicy;

template <typename TPolicyCont, typename...TSublayers>
struct GetSublayerPolicy<TPolicyCont, SubLayerContainer<TSublayers...>>
{
    template <typename TRes, typename...T>
    struct imp
    {
        using type = TRes;
    };

    template <typename...T1, typename TCur, typename...T2>
    struct imp<SublayerPolicyContainer<T1...>, TCur, T2...>
    {
        using CurTag = typename TCur::Tag;
        template <typename T>
        using CurLayer = typename TCur::template Layer<T>;
        using CurPolicy = SubPolicyPicker<TPolicyCont, CurTag>;
        using tmp = SublayerPolicyContainer<T1..., SublayerPolicies<CurTag, CurLayer, CurPolicy>>;
        using type = typename imp<tmp, T2...>::type;
    };

    using type = typename imp<SublayerPolicyContainer<>, TSublayers...>::type;
};

template <bool plainFBO, typename TInsts>
struct FeedbackOutCheck
{
    constexpr static bool value = true;
};

template <typename...TInstElements>
struct FeedbackOutCheck<true, SublayerPolicyContainer<TInstElements...>>
{
    template <typename...T>
    struct imp
    {
        constexpr static bool value = true;
    };

    template <typename TTag, template <typename> class TLayer, typename TCurPolicy, typename...T>
    struct imp<SublayerPolicies<TTag, TLayer, TCurPolicy>, T...>
    {
        constexpr static bool tmp = PolicySelect<FeedbackPolicy, TCurPolicy>::IsFeedbackOutput;
        constexpr static bool value = AndValue<tmp, imp<T...>>;
    };

    constexpr static bool value = imp<TInstElements...>::value;
};

namespace NSFeedbackOutSet
{
template <bool isUpdated, typename TTag, typename TRes, typename TInstCont>
struct UpdateByTag
{
    using type = TInstCont;
};

template <typename TTag, typename TRes, typename TInstCont>
struct UpdateByTag<true, TTag, TRes, TInstCont>
{
    using type = TRes;
};

template <typename TTag, typename...TInstRes, typename TCur, typename...TInstRemain>
struct UpdateByTag<true, TTag, SublayerPolicyContainer<TInstRes...>,
                   SublayerPolicyContainer<TCur, TInstRemain...>>
{
    using OriTag = typename TCur::Tag;
    using OriPolicy = typename TCur::Policy;
    template <typename T>
    using OriLayer = typename TCur::template Layer<T>;

    constexpr static bool checkOK = std::is_same<TTag, OriTag>::value;
    using NewPolicy = typename std::conditional_t<checkOK,
                                                  ChangePolicy_<PFeedbackOutput, OriPolicy>,
                                                  Identity_<OriPolicy>>::type;

    using TModified = SublayerPolicies<OriTag, OriLayer, NewPolicy>;
    using NewRes = SublayerPolicyContainer<TInstRes..., TModified>;
    using type = typename UpdateByTag<true, TTag, NewRes, SublayerPolicyContainer<TInstRemain...>>::type;
};

template <typename TTag, typename TInterConnects, typename TInstCont>
struct UpdateBySourceLayer
{
    static_assert(ArraySize<TInterConnects> == 0, "Test Error");
    using type = TInstCont;
};

template <typename TTag, typename TI, typename...TIRemain, typename TInstCont>
struct UpdateBySourceLayer<TTag, InterConnectContainer<TI, TIRemain...>, TInstCont>
{
    using OutTag = typename TI::OutTag;
    using InTag = typename TI::InTag;
    constexpr static bool isOutTag = std::is_same<TTag, OutTag>::value;
    using tmp = typename UpdateByTag<isOutTag, InTag,
                                     SublayerPolicyContainer<>, TInstCont>::type;
    using type = typename UpdateBySourceLayer<TTag, InterConnectContainer<TIRemain...>, tmp>::type;
};
}

template <typename TInsts, typename InterConnects>
struct FeedbackOutSet;

template <typename...TInstElements, typename InterConnects>
struct FeedbackOutSet<SublayerPolicyContainer<TInstElements...>, InterConnects>
{
    template <typename TRes, typename TNotProcessed>
    struct imp
    {
        using type = TRes;
    };

    template <typename...TProcessedInst, typename TCur, typename...TInsts>
    struct imp<SublayerPolicyContainer<TProcessedInst...>,
               SublayerPolicyContainer<TCur, TInsts...>>
    {
        using Tag = typename TCur::Tag;
        using CurPolicies = typename TCur::Policy;

        constexpr static bool isUpdate = PolicySelect<FeedbackPolicy, CurPolicies>::IsFeedbackOutput || \
                                         PolicySelect<FeedbackPolicy, CurPolicies>::IsUpdate;

        using tmp1 = SublayerPolicyContainer<TProcessedInst..., TCur>;
        using tmp2
            = typename std::conditional_t<isUpdate,
                                          NSFeedbackOutSet::UpdateBySourceLayer<Tag, InterConnects,
                                                                                SublayerPolicyContainer<TInsts...>>,
                                          Identity_<SublayerPolicyContainer<TInsts...>>>::type;
        using type = typename imp<tmp1, tmp2>::type;
    };

    using type = typename imp<SublayerPolicyContainer<>, SublayerPolicyContainer<TInstElements...>>::type;
};

template <typename TInsts, typename TSublayerPolicies>
struct Instantiation
{
    using type = TInsts;
};

template <typename...TInsts, typename TCur, typename...TSublayers>
struct Instantiation<InstantiatedSublayerContainer<TInsts...>,
                     SublayerPolicyContainer<TCur, TSublayers...>>
{
    using Tag = typename TCur::Tag;
    using Policy = typename TCur::Policy;

    template <typename T>
    using Layer = typename TCur::template Layer<T>;

    using InstLayer = Layer<Policy>;

    using tmpRes = InstantiatedSublayerContainer<TInsts..., InstantiatedSublayer<Tag, InstLayer>>;
    using type = typename Instantiation<tmpRes, SublayerPolicyContainer<TSublayers...>>::type;
};
}
template <typename TPolicyCont, typename OrderedSublayers, typename InterConnects>
struct SublayerInstantiation
{
    static_assert(IsPolicyContainer<TPolicyCont>, "Not a Policy Container");

    using SublayerWithPolicy = typename NSSI::GetSublayerPolicy<TPolicyCont, OrderedSublayers>::type;

    /// check -- if feedbackout is set in parent layer, then each sublayer should also set it to true
    using PlainPolicies = PlainPolicy<TPolicyCont>;
    constexpr static bool IsPlainPolicyFeedbackOut = PolicySelect<FeedbackPolicy, PlainPolicies>::IsFeedbackOutput;
    static_assert(NSSI::FeedbackOutCheck<IsPlainPolicyFeedbackOut, SublayerWithPolicy>::value,
                                            "Sublayer not set feedback output, logic error!");

    /// for any instance A, if there is a connection A->B and A is feedbackin, then B should set to feedbackout
    using FeedbackOutUpdate = typename std::conditional_t<IsPlainPolicyFeedbackOut,
                                                 Identity_<SublayerWithPolicy>,
                                                 NSSI::FeedbackOutSet<SublayerWithPolicy, InterConnects>>::type;
    /// Instantiation
    using type = typename NSSI::Instantiation<InstantiatedSublayerContainer<>, FeedbackOutUpdate>::type;
};

/// ========= Sublayer Tuple Type ======================================================
template <typename...TInst>
struct SublayerTupleType
{
    template <typename TTuple, typename TI, typename...TTags>
    struct imp
    {
        using InstTuple = TTuple;
        using TagTuple = std::tuple<TTags...>;
    };

    template <typename...T1, typename TCur, typename...T2, typename...T3>
    struct imp<std::tuple<T1...>, InstantiatedSublayerContainer<TCur, T2...>, T3...>
    {
        using CurTag = typename TCur::Tag;
        using tmp = std::tuple<T1..., typename TCur::Layer>;
        using NextStep = imp<tmp, InstantiatedSublayerContainer<T2...>, T3..., CurTag>;
        using InstTuple = typename NextStep::InstTuple;
        using TagTuple = typename NextStep::TagTuple;
    };

    using InstTuple = typename imp<std::tuple<>, InstantiatedSublayerContainer<TInst...>>::InstTuple;
    using TagTuple = typename imp<std::tuple<>, InstantiatedSublayerContainer<TInst...>>::TagTuple;
};

/// ========= Sublayer Processor =======================================================
namespace NSSP
{
template <typename TTag, typename TI> struct Tag2ID;

template <typename TTag, typename...TI>
struct Tag2ID<TTag, std::tuple<TI...>>
{
    constexpr static size_t value = NSNamedParams::Tag2ID<TTag, TI...>;
};

template <typename> struct InternalResultHelper;
template <typename...T>
struct InternalResultHelper<std::tuple<T...>>
{
    using type = NamedParams<T...>;
};

class SublayerBase
{
public:
    virtual ~SublayerBase() = default;
};

template <typename TLayer>
class SublayerDerive : public SublayerBase
{
public:
    SublayerDerive(TLayer layer)
        : m_layer(std::move(layer)) {}

    TLayer& layer()
    {
        return m_layer;
    }

    const TLayer& layer() const
    {
        return m_layer;
    }
private:
    TLayer m_layer;
};

class SublayerPtr
{
public:
    SublayerPtr(SublayerBase* pptr = nullptr)
        : ptr(pptr) {}

    template <typename TLayer>
    SublayerPtr(TLayer&& layer)
    {
        using RawLayer = std::decay_t<TLayer>;
        ptr = std::unique_ptr<SublayerBase>(new SublayerDerive<RawLayer>(std::forward<TLayer>(layer)));
    }

    auto* get()
    {
        return ptr.get();
    }

    const auto* get() const
    {
        return ptr.get();
    }
private:
    std::shared_ptr<SublayerBase> ptr;
};

template <typename TagTuple, typename SublayerTuple>
struct SublayerCreateImp
{
    using SublayerArray = std::vector<SublayerPtr>;
public:
    SublayerCreateImp(size_t s)
        : m_tuple(s) {}

public:
    template <typename TTag, typename...TParams>
    auto Set(TParams&&... params)
    {
        constexpr static size_t TagPos = NSSP::Tag2ID<TTag, TagTuple>::value;
        using AimType = typename std::tuple_element<TagPos, SublayerTuple>::type;
        AimType aimInst(std::forward<TParams>(params)...);
        m_tuple[TagPos] = NSSP::SublayerPtr(std::move(aimInst));
        return *this;
    }

    operator SublayerArray() const
    {
        return m_tuple;
    }

private:
    SublayerArray m_tuple;
};

/// feed forward
namespace NSFeedForward
{
template <typename TAimTag, typename TInputConnects>
struct CreateInputFromInConnect
{
    template <typename TInput, typename TRes>
    static auto Eval(const TInput&, TRes&& res)
    {
        return std::forward<TRes>(res);
    }
};

template <typename TAimTag, typename TCur, typename...TI>
struct CreateInputFromInConnect<TAimTag, InConnectContainer<TCur, TI...>>
{
    constexpr static bool isUseful = std::is_same<TAimTag, typename TCur::InLayerTag>::value;
    template <bool checkRes, typename TDummy = void>
    struct imp
    {
        template <typename TInput, typename TRes>
        static auto Eval(const TInput&, TRes&& res)
        {
            return std::forward<TRes>(res);
        }
    };

    template <typename TDummy>
    struct imp<true, TDummy>
    {
        template <typename TInput, typename TRes>
        static auto Eval(const TInput& input, TRes&& res)
        {
            using InName = typename TCur::InName;
            using InLayerName = typename TCur::InLayerName;
            return std::forward<TRes>(res).template Set<InLayerName>(input.template Get<InName>());
        }
    };

    template <typename TInput, typename TRes>
    static auto Eval(const TInput& input, TRes&& res)
    {
        auto cur = imp<isUseful>::Eval(input, std::forward<TRes>(res));
        using NextStep = CreateInputFromInConnect<TAimTag, InConnectContainer<TI...>>;
        return NextStep::Eval(input, std::move(cur));
    }
};

template <typename TAimTag, typename TInternalConnects>
struct CreateInputFromInternalConnect
{
    template <typename TInternal, typename TRes>
    static auto Eval(const TInternal&, TRes&& res)
    {
        return std::forward<TRes>(res);
    }
};

template <typename TAimTag, typename TCur, typename...TI>
struct CreateInputFromInternalConnect<TAimTag, InterConnectContainer<TCur, TI...>>
{
    constexpr static bool isUseful = std::is_same<TAimTag, typename TCur::InTag>::value;
    template <bool checkRes, typename TDummy = void>
    struct imp
    {
        template <typename TInternal, typename TRes>
        static auto Eval(const TInternal&, TRes&& res)
        {
            return std::forward<TRes>(res);
        }
    };

    template <typename TDummy>
    struct imp<true, TDummy>
    {
        template <typename TInternal, typename TRes>
        static auto Eval(const TInternal& input, TRes&& res)
        {
            using OutTag = typename TCur::OutTag;
            using OutName = typename TCur::OutName;
            using InName = typename TCur::InName;
            auto preLayer = input.template Get<OutTag>();

            return std::forward<TRes>(res).template Set<InName>(preLayer.template Get<OutName>());
        }
    };

    template <typename TInternal, typename TRes>
    static auto Eval(const TInternal& input, TRes&& res)
    {
        auto cur = imp<isUseful>::Eval(input, std::forward<TRes>(res));

        using NextStep = CreateInputFromInternalConnect<TAimTag, InterConnectContainer<TI...>>;
        return NextStep::Eval(input, std::move(cur));
    }
};

template <typename TAimTag, typename TOutputConnects>
struct FillOutput
{
    template <typename TRes, typename TO>
    static auto Eval(const TRes&, TO&& output)
    {
        return std::forward<TO>(output);
    }
};

template <typename TAimTag, typename TCur, typename...TI>
struct FillOutput<TAimTag, OutConnectContainer<TCur, TI...>>
{
    constexpr static bool isUseful = std::is_same<TAimTag, typename TCur::OutLayerTag>::value;
    template <bool checkRes, typename TDummy = void>
    struct imp
    {
        template <typename TOutput, typename TRes>
        static auto Eval(const TRes&, TOutput&& output)
        {
            return std::forward<TOutput>(output);
        }
    };

    template <typename TDummy>
    struct imp<true, TDummy>
    {
        template <typename TOutput, typename TRes>
        static auto Eval(const TRes& curLayerRes, TOutput&& output)
        {
            using OutLayerName = typename TCur::OutLayerName;
            using OutName = typename TCur::OutName;

            auto tmp = curLayerRes.template Get<OutLayerName>();
            return std::forward<TOutput>(output).template Set<OutName>(std::move(tmp));
        }
    };

    template <typename TRes, typename TO>
    static auto Eval(const TRes& curLayerRes, TO&& output)
    {
        auto cur = imp<isUseful>::Eval(curLayerRes, std::forward<TO>(output));

        using NextStep = FillOutput<TAimTag, OutConnectContainer<TI...>>;
        return NextStep::Eval(curLayerRes, std::move(cur));
    }
};
}

template <typename TSublayerTuple, typename TagTuple,
          typename TInputConnects, typename TOutputConnects, typename TInnerConnects>
struct FeedForward
{
    template <typename TInput, typename SublayerTupleIt, typename TInternal, typename TOutput>
    static auto Eval(const TInput&, SublayerTupleIt&&, const TInternal&, TOutput&& output)
    {
        return std::forward<TOutput>(output);
    }
};

template <typename TSublayer, typename...TRemainSublayers,
          typename AimTag, typename...TRemainTags,
          typename TInputConnects, typename TOutputConnects, typename TInnerConnects>
struct FeedForward<std::tuple<TSublayer, TRemainSublayers...>,
                   std::tuple<AimTag, TRemainTags...>,
                   TInputConnects, TOutputConnects, TInnerConnects>
{
    template <typename TInput, typename SublayerTupleIt, typename TInternal, typename TOutput>
    static auto Eval(const TInput& input, SublayerTupleIt&& it,
                     TInternal&& internal, TOutput&& output)
    {
        using namespace NSFeedForward;

        auto& curLayer = (static_cast<SublayerDerive<TSublayer>*>(it->get()))->layer();

        using TSublayerInput = typename TSublayer::InputType;
        auto input1 = CreateInputFromInConnect<AimTag, TInputConnects>::Eval(input, TSublayerInput::Create());
        auto input2 = CreateInputFromInternalConnect<AimTag, TInnerConnects>::Eval(internal, std::move(input1));

        auto res = curLayer.FeedForward(std::move(input2));
        auto new_output = FillOutput<AimTag, TOutputConnects>::Eval(res, std::forward<TOutput>(output));
        auto new_internal = std::forward<TInternal>(internal).template Set<AimTag>(std::move(res));
        using NextStep = FeedForward<std::tuple<TRemainSublayers...>,
                                     std::tuple<TRemainTags...>,
                                     TInputConnects, TOutputConnects, TInnerConnects>;
        return NextStep::Eval(input, ++it, std::move(new_internal), std::move(new_output));
    }
};

/// feed backward
namespace NSFeedBackward
{
template <typename TOriPara, typename TNewPara,
          std::enable_if_t<std::is_same<std::decay_t<TOriPara>, NullParameter>::value>* = nullptr>
auto ParaMerge(TOriPara&&, TNewPara&& p_new)
{
    return std::forward<TNewPara>(p_new);
}

template <typename TOriPara, typename TNewPara,
          std::enable_if_t<!std::is_same<std::decay_t<TOriPara>, NullParameter>::value>* = nullptr>
auto ParaMerge(TOriPara&& ori, TNewPara&& p_new)
{
    return std::forward<TOriPara>(ori) + std::forward<TNewPara>(p_new);
}

template <typename TAimTag, typename TOutputConnects>
struct CreateGradFromOutside
{
    template <typename TGrad, typename TRes>
    static auto Eval(const TGrad&, const TRes& res)
    {
        return res;
    }
};

template <typename TAimTag, typename TCur, typename...TI>
struct CreateGradFromOutside<TAimTag, OutConnectContainer<TCur, TI...>>
{
    constexpr static bool isUseful = std::is_same<TAimTag, typename TCur::OutLayerTag>::value;
    template <bool checkRes, typename TDummy = void>
    struct imp
    {
        template <typename TGrad, typename TRes>
        static auto Eval(const TGrad&, TRes&& res)
        {
            return std::forward<TRes>(res);
        }
    };

    template <typename TDummy>
    struct imp<true, TDummy>
    {
        template <typename TGrad, typename TRes>
        static auto Eval(const TGrad& grad, TRes&& res)
        {
            using OutLayerName = typename TCur::OutLayerName;
            using OutName = typename TCur::OutName;

            auto tmp = grad.template Get<OutName>();
            auto merged_res = ParaMerge(res.template Get<OutLayerName>(), tmp);
            return std::forward<TRes>(res).template Set<OutLayerName>(merged_res);
        }
    };

    template <typename TGrad, typename TRes>
    static auto Eval(const TGrad& grad, TRes&& res)
    {
        auto cur = imp<isUseful>::Eval(grad, std::forward<TRes>(res));

        using NextStep = CreateGradFromOutside<TAimTag, OutConnectContainer<TI...>>;
        return NextStep::Eval(grad, std::move(cur));
    }
};

template <typename TAimTag, typename TInternalConnects>
struct CreateGradFromInternal
{
    template <typename TInternal, typename TRes>
    static auto Eval(const TInternal&, TRes&& res)
    {
        return std::forward<TRes>(res);
    }
};

template <typename TAimTag, typename TCur, typename...TI>
struct CreateGradFromInternal<TAimTag, InterConnectContainer<TCur, TI...>>
{
    constexpr static bool isUseful = std::is_same<TAimTag, typename TCur::OutTag>::value;
    template <bool checkRes, typename TDummy = void>
    struct imp
    {
        template <typename TInternal, typename TRes>
        static auto Eval(const TInternal&, TRes&& res)
        {
            return std::forward<TRes>(res);
        }
    };

    template <typename TDummy>
    struct imp<true, TDummy>
    {
        template <typename TInternal, typename TRes>
        static auto Eval(const TInternal& input, TRes&& res)
        {
            using InTag = typename TCur::InTag;
            using OutName = typename TCur::OutName;
            using InName = typename TCur::InName;
            auto postLayer = input.template Get<InTag>();

            auto tmp = postLayer.template Get<InName>();
            auto merged_res = ParaMerge(res.template Get<OutName>(), tmp);
            return std::forward<TRes>(res).template Set<OutName>(merged_res);
        }
    };

    template <typename TInternal, typename TRes>
    static auto Eval(const TInternal& input, TRes&& res)
    {
        auto cur = imp<isUseful>::Eval(input, std::forward<TRes>(res));

        using NextStep = CreateGradFromInternal<TAimTag, InterConnectContainer<TI...>>;
        return NextStep::Eval(input, std::move(cur));
    }
};

template <typename TAimTag, typename TInputConnects>
struct FillResult
{
    template <typename TNewInternal, typename TNewInput>
    static auto Eval(const TNewInternal&, TNewInput&& res)
    {
        return std::forward<TNewInput>(res);
    }
};

template <typename TAimTag, typename TCur, typename...TI>
struct FillResult<TAimTag, InConnectContainer<TCur, TI...>>
{
    constexpr static bool isUseful = std::is_same<TAimTag, typename TCur::InLayerTag>::value;
    template <bool checkRes, typename TDummy = void>
    struct imp
    {
        template <typename TNewInternal, typename TNewInput>
        static auto Eval(const TNewInternal&, TNewInput&& res)
        {
            return std::forward<TNewInput>(res);
        }
    };

    template <typename TDummy>
    struct imp<true, TDummy>
    {
        template <typename TNewInternal, typename TNewInput>
        static auto Eval(const TNewInternal& curInternal, TNewInput&& res)
        {
            using InName = typename TCur::InName;
            using InLayerName = typename TCur::InLayerName;

            auto tmp = curInternal.template Get<InLayerName>();
            auto merged_res = ParaMerge(res.template Get<InName>(), tmp);
            return std::forward<TNewInput>(res).template Set<InName>(merged_res);
        }
    };

    template <typename TNewInternal, typename TNewInput>
    static auto Eval(const TNewInternal& curInternal, TNewInput&& res)
    {
        auto cur = imp<isUseful>::Eval(curInternal, std::forward<TNewInput>(res));

        using NextStep = FillResult<TAimTag, InConnectContainer<TI...>>;
        return NextStep::Eval(curInternal, std::move(cur));
    }
};
}

template <typename TRevSublayerTuple, typename TRevTagTuple,
          typename InputConnects, typename InterConnects, typename OutputConnects>
struct FeedBackward
{
    template <typename TGrad, typename SublayerTupleIt, typename TInternal, typename TInput>
    static auto Eval(const TGrad&, SublayerTupleIt&&, const TInternal&, TInput&& res)
    {
        return std::forward<TInput>(res);
    }
};

template <typename TSublayer, typename...TSublayerRemain,
          typename AimTag, typename...TAimTagRemain,
          typename InputConnects, typename InterConnects, typename OutputConnects>
struct FeedBackward<std::tuple<TSublayer, TSublayerRemain...>,
                    std::tuple<AimTag, TAimTagRemain...>,
                    InputConnects, InterConnects, OutputConnects>
{
    template <typename TGrad, typename SublayerTupleIt, typename TInternal, typename TInput>
    static auto Eval(const TGrad& grad, SublayerTupleIt&& it,
                     TInternal&& internal, TInput&& res)
    {
        using namespace NSFeedBackward;
        auto& curLayer = (static_cast<SublayerDerive<TSublayer>*>(it->get()))->layer();
        using TSublayerOutput = typename TSublayer::OutputType;

        auto grad1 = CreateGradFromOutside<AimTag, OutputConnects>::Eval(grad, TSublayerOutput::Create());
        auto grad2 = CreateGradFromInternal<AimTag, InterConnects>::Eval(internal, std::move(grad1));

        auto curLayerRes = curLayer.FeedBackward(std::move(grad2));
        auto new_res = FillResult<AimTag, InputConnects>::Eval(curLayerRes, std::forward<TInput>(res));
        auto new_internal = std::forward<TInternal>(internal).template Set<AimTag>(std::move(curLayerRes));

        using NextStep = FeedBackward<std::tuple<TSublayerRemain...>,
                                      std::tuple<TAimTagRemain...>, InputConnects, InterConnects, OutputConnects>;
        return NextStep::Eval(grad, ++it, std::move(new_internal), std::move(new_res));
    }
};

/// other operations
template <typename TCont>
struct Init
{
    template <typename TInitializer, typename TLoader, typename TIter>
    static auto Eval(TInitializer&, TLoader&, TIter&&) {}
};

template <template <typename...> class TCont, typename TCur, typename...TRemain>
struct Init<TCont<TCur, TRemain...>>
{
    template <typename TInitializer, typename TLoader, typename TIter>
    static auto Eval(TInitializer& initializer, TLoader& loader, TIter&& it)
    {
        auto& curLayer = (static_cast<SublayerDerive<TCur>*>(it->get()))->layer();
        LayerInit(curLayer, initializer, loader);
        using NextStep = Init<TCont<TRemain...>>;
        NextStep::Eval(initializer, loader, ++it);
    }
};

template <typename TCont>
struct GradCollect
{
    template <typename TGradCollector, typename TIter>
    static auto Eval(TGradCollector&, TIter&&) {}
};

template <template <typename...> class TCont, typename TCur, typename...TRemain>
struct GradCollect<TCont<TCur, TRemain...>>
{
    template <typename TGradCollector, typename TIter>
    static auto Eval(TGradCollector& collector, TIter&& it)
    {
        auto& curLayer = (static_cast<SublayerDerive<TCur>*>(it->get()))->layer();
        LayerGradCollect(curLayer, collector);
        using NextStep = GradCollect<TCont<TRemain...>>;
        NextStep::Eval(collector, ++it);
    }
};

template <typename TCont>
struct LoadWeights
{
    template <typename TLoad, typename TIter>
    static auto Eval(const TLoad&, TIter&&) {}
};

template <template <typename...> class TCont, typename TCur, typename...TRemain>
struct LoadWeights<TCont<TCur, TRemain...>>
{
    template <typename TLoad, typename TIter>
    static auto Eval(const TLoad& loader, TIter&& it)
    {
        auto& curLayer = (static_cast<SublayerDerive<TCur>*>(it->get()))->layer();
        LayerLoadWeights(curLayer, loader);
        using NextStep = LoadWeights<TCont<TRemain...>>;
        NextStep::Eval(loader, ++it);
    }
};

template <typename TCont>
struct SaveWeights
{
    template <typename TSave, typename TIter>
    static auto Eval(TSave&, TIter&&) {}
};

template <template <typename...> class TCont, typename TCur, typename...TRemain>
struct SaveWeights<TCont<TCur, TRemain...>>
{
    template <typename TSave, typename TIter>
    static auto Eval(TSave& saver, TIter&& it)
    {
        auto& curLayer = (static_cast<const SublayerDerive<TCur>*>(it->get()))->layer();
        LayerSaveWeights(curLayer, saver);
        using NextStep = SaveWeights<TCont<TRemain...>>;
        NextStep::Eval(saver, ++it);
    }
};

template <typename TCont>
struct NeutralInvariant
{
    template <typename TSave, typename TIter>
    static auto Eval(TSave&, TIter&&) {}
};

template <template <typename...> class TCont, typename TCur, typename...TRemain>
struct NeutralInvariant<TCont<TCur, TRemain...>>
{
    template <typename TIter>
    static auto Eval(TIter&& it)
    {
        auto& curLayer = (static_cast<SublayerDerive<TCur>*>(it->get()))->layer();
        LayerNeutralInvariant(curLayer);
        using NextStep = NeutralInvariant<TCont<TRemain...>>;
        NextStep::Eval(++it);
    }
};
}

template <typename TInstContainer, typename InputConnects, typename OutputConnects, typename InterConnects>
class SublayerProcessor;

template <typename...TInst, typename InputConnects, typename OutputConnects, typename InterConnects>
class SublayerProcessor<NSComposeKernel::InstantiatedSublayerContainer<TInst...>,
                        InputConnects, OutputConnects, InterConnects>
{
private:
    using SublayerTuple = typename SublayerTupleType<TInst...>::InstTuple;
    using TagTuple = typename SublayerTupleType<TInst...>::TagTuple;
    using RevSublayerTuple = LayerTraits::Reverse<SublayerTuple>;
    using RevTagTuple = LayerTraits::Reverse<TagTuple>;
    using InternalResult = typename NSSP::InternalResultHelper<TagTuple>::type;

public:
    using SublayerArray = typename NSSP::template SublayerCreateImp<TagTuple, SublayerTuple>::SublayerArray;

    static auto Create()
    {
        return NSSP::template SublayerCreateImp<TagTuple, SublayerTuple>(ArraySize<TagTuple>);
    }

    template <typename TInput, typename TOutput>
    static auto Feedforward(const TInput& input, SublayerArray& sublayers, const TOutput& output)
    {
        using TCal = NSSP::FeedForward<SublayerTuple, TagTuple, InputConnects, OutputConnects, InterConnects>;
        return TCal::Eval(input, sublayers.begin(), InternalResult::Create(), output);
    }

    template <typename TGrad, typename TRes>
    static auto Feedbackward(const TGrad& grad, SublayerArray& sublayers, const TRes& res)
    {
        using TCal = NSSP::FeedBackward<RevSublayerTuple, RevTagTuple, InputConnects, InterConnects, OutputConnects>;
        return TCal::Eval(grad, sublayers.rbegin(), InternalResult::Create(), res);
    }

    template <typename TInitializer, typename TLoader>
    static void Init(TInitializer& initializer, TLoader& loader, SublayerArray& sublayers)
    {
        NSSP::Init<SublayerTuple>::Eval(initializer, loader, sublayers.begin());
    }

    template <typename TGradCollector>
    static void GradCollect(TGradCollector& gc, SublayerArray& sublayers)
    {
        NSSP::GradCollect<SublayerTuple>::Eval(gc, sublayers.begin());
    }

    template <typename TLoad>
    static void LoadWeights(const TLoad& loader, SublayerArray& sublayers)
    {
        NSSP::LoadWeights<SublayerTuple>::Eval(loader, sublayers.begin());
    }

    template <typename TSave>
    static void SaveWeights(TSave& saver, const SublayerArray& sublayers)
    {
        NSSP::SaveWeights<SublayerTuple>::Eval(saver, sublayers.begin());
    }

    static void NeutralInvariant(SublayerArray& sublayers)
    {
        NSSP::NeutralInvariant<SublayerTuple>::Eval(sublayers.begin());
    }
};
}

template <typename...TComposeKernelInfo>
struct ComposeTopology
{
/// ========== Separate Results ========================================
    using SubLayers = typename NSComposeKernel::SeparateParameters<TComposeKernelInfo...>::SubLayerRes;
    using InterConnects = typename NSComposeKernel::SeparateParameters<TComposeKernelInfo...>::InterConnectRes;
    using InputConnects = typename NSComposeKernel::SeparateParameters<TComposeKernelInfo...>::InConnectRes;
    using OutputConnects = typename NSComposeKernel::SeparateParameters<TComposeKernelInfo...>::OutConnectRes;

/// ========== Asserts =================================================
    static_assert((ArraySize<SubLayers> != 0), "Sublayer is empty.");
    static_assert((NSComposeKernel::SublayerCheck<SubLayers>::IsUnique), "Two or more sublayers have same tag.");
    static_assert((NSComposeKernel::InternalConnectCheck<InterConnects>::NoSelfCycle), "Internal connections have self-connect.");
    static_assert((NSComposeKernel::InternalConnectCheck<InterConnects>::UniqueSource), "One internal input corresponds to two or more internal outputs.");
    static_assert((NSComposeKernel::InputConnectCheck<InputConnects>::UniqueSource), "One input corresponds to two or more sources.");
    static_assert((NSComposeKernel::OutputConnectCheck<OutputConnects>::UniqueSource), "One output corresponds to two or more sources.");
    static_assert((NSComposeKernel::InternalTagInSublayer<InterConnects, SubLayers>::value), "Internal connections have tags are not sublayer tags.");
    static_assert((NSComposeKernel::InputTagInSubLayer<InputConnects, SubLayers>::value), "One or more input tags are not sublayer tags.");
    static_assert((NSComposeKernel::OutputTagInSubLayer<OutputConnects, SubLayers>::value), "One or more output tags are not sublayer tags.");
    static_assert((NSComposeKernel::SublayerTagInOtherArrays<InterConnects, InputConnects,
                                            OutputConnects, SubLayers>::value), "One ore more sublayer tags not belong to any connection containers.");
    static_assert((NSComposeKernel::UsefulInternalPostLayer<InterConnects, OutputConnects>::value), "Internal output info is useless");
    static_assert((NSComposeKernel::UsefulInputLayer<InputConnects, InterConnects, OutputConnects>::value), "Input info is useless");

/// ========== Topological Ordering ===================================
    using TopologicalOrdering = typename NSComposeKernel::TopologicalOrdering_<SubLayers, InterConnects>::type;

    template <typename TPolicyCont>
    using Instances = typename NSComposeKernel::SublayerInstantiation<TPolicyCont, TopologicalOrdering, InterConnects>::type;

    template <typename TPolicyCont>
    using SublayerTupleCreator = NSComposeKernel::SublayerProcessor<Instances<TPolicyCont>,
                                                                    InputConnects,
                                                                    OutputConnects,
                                                                    InterConnects>;

/// ========= Update Check =============================================================
    template <typename TPolicyCont>
    struct IsUpdate
    {
        template <typename TI>
        struct imp
        {
            constexpr static bool value = false;
        };

        template <typename TCur, typename...TInsts>
        struct imp<NSComposeKernel::InstantiatedSublayerContainer<TCur, TInsts...>>
        {
            constexpr static bool tmp = TCur::Layer::IsUpdate;
            constexpr static bool value = OrValue<tmp, imp<NSComposeKernel::InstantiatedSublayerContainer<TInsts...>>>;
        };
        constexpr static bool value = imp<Instances<TPolicyCont>>::value;
    };
};

template <typename TInputType, typename TOutputType, typename TPolicyCont, typename TKernelTopo>
class ComposeKernel
{
    static_assert(IsPolicyContainer<TPolicyCont>, "Parameter is not a policy container.");
    using PlainPolicies = PlainPolicy<TPolicyCont>;

protected:
    using TupleProcessor = typename TKernelTopo::template SublayerTupleCreator<TPolicyCont>;
public:
    static constexpr bool IsFeedbackOutput = PolicySelect<FeedbackPolicy, PlainPolicies>::IsFeedbackOutput;
    static constexpr bool IsUpdate = TKernelTopo::template IsUpdate<TPolicyCont>::value;

    using InputType = TInputType;
    using OutputType = TOutputType;

public:
    ComposeKernel(typename TupleProcessor::SublayerArray p_sublayers)
        : sublayers(std::move(p_sublayers)) {}

public:
    template <typename TInitializer, typename TLoad>
    void Init(TInitializer& initializer, TLoad& loader)
    {
        TupleProcessor::Init(initializer, loader, sublayers);
    }

    template <typename TLoad>
    void LoadWeights(const TLoad& loader)
    {
        TupleProcessor::LoadWeights(loader, sublayers);
    }

    template <typename TSave>
    void SaveWeights(TSave& saver) const
    {
        TupleProcessor::SaveWeights(saver, sublayers);
    }

    template <typename TIn>
    auto FeedForward(const TIn& p_in)
    {
        return TupleProcessor::Feedforward(p_in, sublayers, OutputType::Create());
    }

    template <typename TGrad>
    auto FeedBackward(const TGrad& p_grad)
    {
        return TupleProcessor::Feedbackward(p_grad, sublayers, InputType::Create());
    }

    template <typename TGradCollector>
    void GradCollect(TGradCollector& col)
    {
        TupleProcessor::GradCollect(col, sublayers);
    }

    void NeutralInvariant()
    {
        TupleProcessor::NeutralInvariant(sublayers);
    }
private:
    typename TupleProcessor::SublayerArray sublayers;
};

template <template<typename> class Layer>
struct Sublayerof;
}
