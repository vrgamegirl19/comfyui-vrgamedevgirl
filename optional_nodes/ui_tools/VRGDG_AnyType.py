# Wildcard socket type (trick from pythongossss): compares equal to every ComfyUI type.
class AnyType(str):
    def __ne__(self, value):
        return False


any_typ = AnyType("*")
