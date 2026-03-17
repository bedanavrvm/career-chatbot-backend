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
from onetdata.admin_views import admin_onet_import as onet_admin_import
from onetdata.admin_views import admin_onet_dashboard as onet_admin_dashboard
from onetdata.admin_views import admin_onet_dq_coverage as onet_admin_dq_coverage
from onetdata.admin_views import admin_onet_mapping_coverage as onet_admin_mapping_coverage
from onetdata.admin_views import admin_onet_mapping_import as onet_admin_mapping_import
from onetdata.admin_views import admin_onet_mapping_manual as onet_admin_mapping_manual
from onetdata.admin_views import admin_onet_mapping_suggest as onet_admin_mapping_suggest
from onetdata.admin_views import admin_onet_snapshot_import as onet_admin_snapshot_import
from onetdata.admin_views import admin_onet_snapshot_generate as onet_admin_snapshot_generate
from onetdata.admin_views import admin_program_field_review as onet_admin_program_field_review
from onetdata.api import api_onet_occupation_detail as onet_api_occupation_detail
from onetdata.api import api_onet_occupations as onet_api_occupations
from onetdata.api import api_onet_recommendations as onet_api_recommendations
from onetdata.mapping_api import api_field_career_path as onet_api_field_career_path
from onetdata.mapping_api import api_field_careers as onet_api_field_careers
from system.views import health as system_health
from system.views import secure_ping as system_secure_ping
from catalog.careers_api import api_program_career_path as catalog_api_program_career_path
from catalog.careers_api import api_program_careers as catalog_api_program_careers

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
    path('api/catalog/programs/<str:program_id_or_code>/', catalog_api_program_detail, name='api_catalog_program_detail'),
    path('api/catalog/institutions/<str:institution_code>', catalog_api_institution_detail, name='api_catalog_institution_detail'),
    path('admin/etl/upload', etl_admin_etl_upload, name='admin_etl_upload'),
    path('admin/etl/process', etl_admin_etl_process, name='admin_etl_process'),
    path('admin/onet', onet_admin_dashboard, name='admin_onet_dashboard'),
    path('admin/onet/import', onet_admin_import, name='admin_onet_import'),
    path('admin/onet/dq/coverage', onet_admin_dq_coverage, name='admin_onet_dq_coverage'),
    path('admin/onet/snapshot/import', onet_admin_snapshot_import, name='admin_onet_snapshot_import'),
    path('admin/onet/snapshot/generate', onet_admin_snapshot_generate, name='admin_onet_snapshot_generate'),
    path('admin/onet/mappings/suggest', onet_admin_mapping_suggest, name='admin_onet_mapping_suggest'),
    path('admin/onet/mappings/import', onet_admin_mapping_import, name='admin_onet_mapping_import'),
    path('admin/onet/mappings/manual', onet_admin_mapping_manual, name='admin_onet_mapping_manual'),
    path('admin/onet/mappings/coverage', onet_admin_mapping_coverage, name='admin_onet_mapping_coverage'),
    path('admin/onet/review/program-fields', onet_admin_program_field_review, name='admin_onet_program_field_review'),
    path('api/onet/occupations', onet_api_occupations, name='api_onet_occupations'),
    path('api/onet/occupations/<str:soc_code>', onet_api_occupation_detail, name='api_onet_occupation_detail'),
    path('api/onet/recommendations', onet_api_recommendations, name='api_onet_recommendations'),
    path('api/onet/fields/<str:field_slug>/careers', onet_api_field_careers, name='api_onet_field_careers'),
    path('api/onet/fields/<str:field_slug>/career-path', onet_api_field_career_path, name='api_onet_field_career_path'),
    path('api/catalog/programs/<str:program_id>/careers', catalog_api_program_careers, name='api_catalog_program_careers'),
    path('api/catalog/programs/<str:program_id>/career-path', catalog_api_program_career_path, name='api_catalog_program_career_path'),
    # Conversations API
    path('api/', include('conversations.urls')),
    path('admin/', admin.site.urls),
]
