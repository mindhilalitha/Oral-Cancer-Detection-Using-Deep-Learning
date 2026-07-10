from django.shortcuts import render
from django.contrib import messages
from users.models import UserRegistrationModel


def admin_login_check(request):
    if request.method == 'POST':
        usrid = request.POST.get('loginid')
        pswd = request.POST.get('password')
        if usrid in ('admin', 'Admin') and pswd in ('admin', 'Admin'):
            return admin_home(request)
        else:
            messages.error(request, 'Invalid admin credentials.')
    return render(request, 'admin_login.html', {})


def admin_home(request):
    from users.models import PredictionResult
    total_users = UserRegistrationModel.objects.count()
    active_users = UserRegistrationModel.objects.filter(status='activated').count()
    pending_users = UserRegistrationModel.objects.filter(status='waiting').count()
    total_scans = PredictionResult.objects.count()
    return render(request, 'admins/admin_home.html', {
        'total_users': total_users,
        'active_users': active_users,
        'pending_users': pending_users,
        'total_scans': total_scans,
    })


def view_registered_users(request):
    data = UserRegistrationModel.objects.all()
    return render(request, 'admins/view_registered_users.html', {'data': data})


def AdminActivaUsers(request):
    if request.method == 'GET':
        id = request.GET.get('uid')
        UserRegistrationModel.objects.filter(id=id).update(status='activated')
        data = UserRegistrationModel.objects.all()
        return render(request, 'admins/view_registered_users.html', {'data': data})
