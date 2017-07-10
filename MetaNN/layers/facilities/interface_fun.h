#pragma once

#include <type_traits>
namespace MetaNN
{
template <typename TLayer, typename TIn>
auto LayerFeedForward(TLayer& layer, TIn&& p_in)
{
    return layer.FeedForward(std::forward<TIn>(p_in));
}

template <typename TLayer, typename TGrad>
auto LayerFeedBackward(TLayer& layer, TGrad&& p_grad)
{
    return layer.FeedBackward(std::forward<TGrad>(p_grad));
}

/// init interface ========================================
namespace NSLayerInterface
{
template <class> class InterChecker;

template <typename L, typename TInitializer, typename TLoad>
static std::true_type InitTest(InterChecker<decltype(&L::template Init<TInitializer, TLoad>)>*);

template <typename L, typename TInitializer, typename TLoad>
static std::false_type InitTest(...);

template <typename L, typename TInitializer, typename TLoad>
constexpr bool InitCheckRes = std::is_same<std::true_type,
                                             decltype(InitTest<L, TInitializer, TLoad>(nullptr))>::value;
}
template <typename TLayer, typename TInitializer, typename TLoad,
          std::enable_if_t<!NSLayerInterface::InitCheckRes<TLayer, TInitializer, TLoad>>* = nullptr>
void LayerInit(TLayer&, TInitializer&, TLoad&){}

template <typename TLayer, typename TInitializer, typename TLoad,
          std::enable_if_t<NSLayerInterface::InitCheckRes<TLayer, TInitializer, TLoad>>* = nullptr>
void LayerInit(TLayer& layer, TInitializer& initer, TLoad& loader)
{
    layer.Init(initer, loader);
}

/// Grad-collect interface ========================================
namespace NSLayerInterface
{
template <typename L, typename TGradCollector>
static std::true_type GradCollectTest(InterChecker<decltype(&L::template GradCollect<TGradCollector>)>*);

template <typename L, typename TGradCollector>
static std::false_type GradCollectTest(...);

template <typename L, typename TGradCollector>
constexpr bool GradCollectCheckRes = std::is_same<std::true_type,
                                             decltype(GradCollectTest<L, TGradCollector>(nullptr))>::value;
}

template <typename TLayer, typename TGradCollector,
          std::enable_if_t<!NSLayerInterface::GradCollectCheckRes<TLayer, TGradCollector>>* = nullptr>
void LayerGradCollect(TLayer&, TGradCollector&) {}

template <typename TLayer, typename TGradCollector,
          std::enable_if_t<NSLayerInterface::GradCollectCheckRes<TLayer, TGradCollector>>* = nullptr>
void LayerGradCollect(TLayer& layer, TGradCollector& gc)
{
    layer.GradCollect(gc);
}

/// Load-weights interface ========================================
namespace NSLayerInterface
{
template <typename L, typename TLoad>
static std::true_type LoadWeightsTest(InterChecker<decltype(&L::template LoadWeights<TLoad>)>*);

template <typename L, typename TLoad>
static std::false_type LoadWeightsTest(...);

template <typename L, typename TLoad>
constexpr bool LoadWeightsCheckRes = std::is_same<std::true_type,
                                             decltype(LoadWeightsTest<L, TLoad>(nullptr))>::value;
}

template <typename TLayer, typename TLoad,
          std::enable_if_t<!NSLayerInterface::LoadWeightsCheckRes<TLayer, TLoad>>* = nullptr>
void LayerLoadWeights(TLayer&, const TLoad&) {}

template <typename TLayer, typename TLoad,
          std::enable_if_t<NSLayerInterface::LoadWeightsCheckRes<TLayer, TLoad>>* = nullptr>
void LayerLoadWeights(TLayer& layer, const TLoad& loader)
{
    layer.LoadWeights(loader);
}

/// Save-weights interface ========================================
namespace NSLayerInterface
{
template <typename L, typename TSave>
static std::true_type SaveWeightsTest(InterChecker<decltype(&L::template SaveWeights<TSave>)>*);

template <typename L, typename TSave>
static std::false_type SaveWeightsTest(...);

template <typename L, typename TSave>
constexpr bool SaveWeightsCheckRes = std::is_same<std::true_type,
                                             decltype(SaveWeightsTest<L, TSave>(nullptr))>::value;
}
template <typename TLayer, typename TSave,
          std::enable_if_t<!NSLayerInterface::SaveWeightsCheckRes<TLayer, TSave>>* = nullptr>
void LayerSaveWeights(const TLayer&, TSave&) {}

template <typename TLayer, typename TSave,
          std::enable_if_t<NSLayerInterface::SaveWeightsCheckRes<TLayer, TSave>>* = nullptr>
void LayerSaveWeights(const TLayer& layer, TSave& saver)
{
    layer.SaveWeights(saver);
}

/// Neural-invariant interface ========================================
namespace NSLayerInterface
{
template <typename L>
static std::true_type NeutralInvariantTest(InterChecker<decltype(&L::NeutralInvariant)>*);

template <typename L>
static std::false_type NeutralInvariantTest(...);

template <typename L>
constexpr bool NeutralInvariantCheckRes = std::is_same<std::true_type,
                                             decltype(NeutralInvariantTest<L>(nullptr))>::value;
}
template <typename TLayer,
          std::enable_if_t<!NSLayerInterface::NeutralInvariantCheckRes<TLayer>>* = nullptr>
void LayerNeutralInvariant(TLayer&) {}

template <typename TLayer,
          std::enable_if_t<NSLayerInterface::NeutralInvariantCheckRes<TLayer>>* = nullptr>
void LayerNeutralInvariant(TLayer& layer)
{
    layer.NeutralInvariant();
}
}
