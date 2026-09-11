#include <ray/distortion.h>


namespace ray
{


namespace distortion
{


std::vector<Direction> DirectionChoices::GetChoices()
{
    return {
        Direction::forward,
        Direction::inverse};
}


std::string DirectionConverter::ToString(Direction direction)
{
    switch (direction)
    {
        case (Direction::forward):
            return "forward";

        case (Direction::inverse):
            return "inverse";

        default:
            throw std::logic_error("Unknown Direction");
    }
}


std::ostream & operator<<(std::ostream &output, Direction value)
{
    return output << DirectionConverter::ToString(value);
}


void to_json(nlohmann::json &json, Direction direction)
{
    json = DirectionConverter::ToString(direction);
}


void from_json(const nlohmann::json &json, Direction &direction)
{
    if (json.is_number_integer())
    {
        auto recovered = Direction(json.get<int>());

        if (recovered == Direction::forward || recovered == Direction::inverse)
        {
            direction = recovered;
        }
        else
        {
            throw std::invalid_argument("Not a valid direction");
        }
    }
    else
    {
        auto asString = json.get<std::string>();
        if (asString == "forward")
        {
            direction = Direction::forward;
        }
        else if (asString == "inverse")
        {
            direction = Direction::inverse;
        }
        else
        {
            throw std::invalid_argument("Not a valid direction");
        }
    }
}


} // end namespace distortion


} // end namespace ray
