from django.shortcuts import render

# Create your views here.
#홈에서 요청하면 jwchathome을 부른다
def home(request):
    context = {}
    return render(request,"jwchathome.html",context)