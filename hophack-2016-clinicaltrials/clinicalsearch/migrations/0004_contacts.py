# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('clinicalsearch', '0003_clinicaltrial_last_changed'),
    ]

    operations = [
        migrations.CreateModel(
            name='Contacts',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('contact_name', models.CharField(max_length=200)),
                ('contact_phone', models.CharField(max_length=15)),
                ('contact_email', models.CharField(max_length=30)),
                ('content', models.TextField()),
            ],
        ),
    ]
