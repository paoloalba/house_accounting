from enum import Enum

class MainCategory(Enum):
    Income = 1
    Outcome = 2

class SubCategory(Enum):
    Initial = 0
    House = 1
    Salary = 2
    Generic = 3
    OneTime = 4
    Exceptional = 6
    Food = 7
    Transport = 8
    Health = 9
    Travel = 10

class TimeCategory(Enum):
    Recurring = 1
    OneTime = 2
    Exceptional = 3

class SampleFrequency(Enum):
    Daily=1
    Weekly=2
    Monthly=3
    Quarterly=4
    Yearly=5