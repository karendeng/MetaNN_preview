#include <iostream>
#include <cassert>
#include <set>
#include <MetaNN/meta_nn.h>
using namespace std;
using namespace MetaNN;

namespace
{
struct Tag1;
struct Tag2;
struct Tag3;

void test_change_policy1()
{
    cout << "Test change policy case 1...\t";
    using input = PolicyContainer<PRowVecInput, PFloatElement,
                                  SubPolicyContainer<Tag1, PColVecInput>>;

    using check1 = ChangePolicy<PEnableBptt, input>;
    static_assert(is_same<check1, PolicyContainer<PRowVecInput, PFloatElement, SubPolicyContainer<Tag1, PColVecInput>, PEnableBptt>>::value, "Check Error");

    using check2 = ChangePolicy<PColVecInput, input>;
    static_assert(is_same<check2, PolicyContainer<PFloatElement, SubPolicyContainer<Tag1, PColVecInput>, PColVecInput>>::value, "Check Error");
    cout << "done" << endl;
}
}
void test_change_policy()
{
    test_change_policy1();
}
