from enum import Enum


class MainCategory(Enum):
    Income = 1
    Outcome = 2


class SubCategory(Enum):
    NA = -1
    Initial = 0
    House = 1
    Salary = 2
    Generic = 3
    Food = 4
    Transport = 5
    Health = 6
    Travel = 7


class TimeCategory(Enum):
    NA = -1
    Recurring = 1
    OneTime = 2
    Exceptional = 3


class SampleFrequency(Enum):
    Daily = 1
    Weekly = 2
    Monthly = 3
    Quarterly = 4
    Yearly = 5
