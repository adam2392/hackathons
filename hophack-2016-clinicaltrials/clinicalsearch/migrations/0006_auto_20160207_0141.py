# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('clinicalsearch', '0005_auto_20160206_2222'),
    ]

    operations = [
        migrations.AddField(
            model_name='clinicaltrial',
            name='ranking',
            field=models.IntegerField(default=1),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='contact',
            name='id',
            field=models.TextField(serialize=False, primary_key=True),
        ),
    ]
