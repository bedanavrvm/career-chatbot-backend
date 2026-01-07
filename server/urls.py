"""
URL configuration for server project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import include, path

from catalog.api import (
    api_catalog_institution_detail as catalog_api_institution_detail,
    api_catalog_program_detail as catalog_api_program_detail,
    api_catalog_status as catalog_api_catalog_status,
    api_program_costs as catalog_api_program_costs,
    api_suffix_mapping as catalog_api_suffix_mapping,
)
from etl.admin_views import admin_etl_process as etl_admin_etl_process
from etl.admin_views import admin_etl_upload as etl_admin_etl_upload
from etl.api import api_eligibility as etl_api_eligibility
from etl.api import api_fields as etl_api_fields
from etl.api import api_institutions as etl_api_institutions
from etl.api import api_programs as etl_api_programs
from etl.api import api_search as etl_api_search
from system.views import health as system_health
from system.views import secure_ping as system_secure_ping

urlpatterns = [
    path('_health/', system_health, name='health'),
    path('_health', system_health, name='health_no_slash'),
    path('api/auth/', include('accounts.urls')),
    path('api/secure-ping/', system_secure_ping, name='secure_ping'),
    path('api/etl/programs', etl_api_programs, name='api_programs'),
    path('api/etl/eligibility', etl_api_eligibility, name='api_eligibility'),
    path('api/etl/institutions', etl_api_institutions, name='api_institutions'),
    path('api/etl/fields', etl_api_fields, name='api_fields'),
    path('api/etl/search', etl_api_search, name='api_search'),
    path('api/catalog/suffix-mapping', catalog_api_suffix_mapping, name='api_suffix_mapping'),
    path('api/catalog/program-costs', catalog_api_program_costs, name='api_program_costs'),
    path('api/catalog/status', catalog_api_catalog_status, name='api_catalog_status'),
    path('api/catalog/programs/<int:program_id>', catalog_api_program_detail, name='api_catalog_program_detail'),
    path('api/catalog/institutions/<str:institution_code>', catalog_api_institution_detail, name='api_catalog_institution_detail'),
    path('admin/etl/upload', etl_admin_etl_upload, name='admin_etl_upload'),
    path('admin/etl/process', etl_admin_etl_process, name='admin_etl_process'),
    # Conversations API
    path('api/', include('conversations.urls')),
    path('admin/', admin.site.urls),
]
