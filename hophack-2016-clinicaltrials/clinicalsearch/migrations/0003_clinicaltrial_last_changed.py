# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('clinicalsearch', '0002_remove_clinicaltrial_last_changed'),
    ]

    operations = [
        migrations.AddField(
            model_name='clinicaltrial',
            name='last_changed',
            field=models.TextField(default=''),
            preserve_default=False,
        ),
    ]
