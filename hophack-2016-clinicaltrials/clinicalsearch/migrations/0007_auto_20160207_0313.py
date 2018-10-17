# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('clinicalsearch', '0006_auto_20160207_0141'),
    ]

    operations = [
        migrations.AddField(
            model_name='clinicaltrial',
            name='genders',
            field=models.TextField(default='Both'),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='clinicaltrial',
            name='health',
            field=models.BooleanField(default=True),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='clinicaltrial',
            name='max_age',
            field=models.IntegerField(default=150),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='clinicaltrial',
            name='min_age',
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
    ]
