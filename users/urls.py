from django.urls import path
from users import views as users_views

urlpatterns = [
    path('user_register_action/', users_views.user_register_action,
         name='user_register_action'),
    path("user_login_check/", users_views.user_login_check, name="user_login_check"),
    path("user_home/", users_views.user_home, name="user_home"),
    path('predict/', users_views.predict_image, name='predict_image'),
    path('train/', users_views.train_model, name='train_model'),
    path('profile/', users_views.user_profile, name='user_profile'),
    path('my-results/', users_views.my_results, name='my_results'),
    path('delete-result/<int:result_id>/', users_views.delete_result, name='delete_result'),
    
]
from django.conf import settings
from django.conf.urls.static import static

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
