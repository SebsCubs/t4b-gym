# import rdflib
from rdflib import Graph, Namespace
import types
import owlready2

def get_object_properties(object_):
    return {key: value for (key, value) in vars(object_).items()}


def get_object_attributes(obj):
    attributes = dir(obj)
    attributes = [attr for attr in attributes if attr[:2]!="__"]#Remove callables
    return attributes

EX = Namespace("https://saref.etsi.org/core/v3.2.1/")

onto = owlready2.get_ontology("https://saref.etsi.org/core/v3.2.1/").load()
# print(dir(onto))
# Device_ = types.new_class("Device_", (onto.Device,))
# Temperature_ = types.new_class("Temperature_", (onto.Temperature,)) 

device = onto.Device()
device.controls = [onto.Temperature()]


print(get_object_properties(device))
print(get_object_attributes(device))

print(device.controls)
print(device.conts)
print()
aaa
for c in onto.classes():
    print("-----")
    print(c)
    inst = c()
    test_class = types.new_class("NewClassName", (c,))


# class Drug():
#     namespace = onto


# g = Graph()

# print(type(EX.device))
# print(dir(EX))


