# from transformers import AutoTokenizer, AutoModel
# tokenizer = AutoTokenizer.from_pretrained("albert/albert-base-v2")
# model = AutoModel.from_pretrained("albert/albert-base-v2")

# tokenizer.save_pretrained('/home/dev/Xavier/LJMU/pre-train/albert')
# model.save_pretrained('/home/dev/Xavier/LJMU/pre-train/albert')



class MedicalA:
    def __init__(self) -> None:
        self.one = 100
        self.two = 300
        
    def get(self):
        self.mul = 22333
        
        
class MedicalCB:
    def __init__(self):
        self.cls = MedicalA()
        
gg = MedicalCB()
gg.cls.get()

print(gg.cls.mul)