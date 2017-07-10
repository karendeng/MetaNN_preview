#pragma once

#include <MetaNN/policies/policy_container.h>
namespace MetaNN
{
template<template <typename> class TLayer, typename...TPolicies>
using MakeLayer = TLayer<PolicyContainer<TPolicies...>>;
}
