#include "ray/pose.h"


namespace ray
{


template struct Pose<float>;
template struct Pose<double>;


} // end namespace ray


template struct pex::Group
    <
        ray::PoseFields,
        ray::PoseTemplate<float>::template Template,
        pex::PlainT<ray::Pose<float>>
    >;


template struct pex::Group
    <
        ray::PoseFields,
        ray::PoseTemplate<double>::template Template,
        pex::PlainT<ray::Pose<double>>
    >;
