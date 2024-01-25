# Generated by Django 4.2 on 2024-01-16 05:31

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('lungspot', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='result',
            name='output_img',
        ),
        migrations.AddField(
            model_name='result',
            name='gradcam_img',
            field=models.ImageField(null=True, upload_to='output-gradcam/'),
        ),
        migrations.AddField(
            model_name='result',
            name='input_img',
            field=models.ImageField(null=True, upload_to='input/'),
        ),
    ]