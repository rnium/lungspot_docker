# Generated by Django 4.2 on 2024-01-14 15:49

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Result',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('output_img', models.ImageField(upload_to='result/')),
                ('prediction', models.CharField(max_length=10)),
                ('created_in', models.DateTimeField(default=django.utils.timezone.now)),
            ],
        ),
    ]
