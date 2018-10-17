# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('clinicalsearch', '0007_auto_20160207_0313'),
    ]

    operations = [
        migrations.RenameField(
            model_name='clinicaltrial',
            old_name='closed',
            new_name='ongoing',
        ),
    ]
