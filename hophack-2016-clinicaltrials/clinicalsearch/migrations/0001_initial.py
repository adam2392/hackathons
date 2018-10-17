# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ClinicalTrial',
            fields=[
                ('id', models.TextField(serialize=False, primary_key=True)),
                ('sponsor', models.TextField()),
                ('published', models.BooleanField()),
                ('state', models.TextField()),
                ('url', models.TextField()),
                ('closed', models.BooleanField()),
                ('title', models.TextField()),
                ('condition', models.TextField()),
                ('intervention', models.TextField()),
                ('locations', models.TextField()),
                ('last_changed', models.TextField()),
            ],
        ),
    ]
