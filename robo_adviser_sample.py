from pydantic import BaseModel

class RoboAdviserSample(BaseModel):
    location: str
    age: float
    smoker: float
    familyMembers: float
    salary: float
    sex: str
    profileType: str

    def __getitem__(self, item):
        return getattr(self, item)
