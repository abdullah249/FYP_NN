from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User

# Create your models here.

class dialysis(models.Model):
	mrno = models.TextField()
	visit_date = models.DateTimeField(default=timezone.now)
	total_dialysis =models.IntegerField()
	dry_wt = models.IntegerField()
	reuse = models.IntegerField()
	bloodtrans = models.IntegerField()
	comordity  = models.TextField()
	heparin = models.IntegerField()
	heparin_infusion = models.IntegerField()
	esrd_cause = models.TextField()
	dialysis_duration = models.IntegerField()
	hbsAg = models.IntegerField()
	antiHcv =models.IntegerField()
	bp = models.IntegerField()
	sbpPre = models.IntegerField()
	sbpPost = models.IntegerField()
	dbpPre = models.IntegerField()
	dbpPost = models.IntegerField()
	pls1 = models.IntegerField()
	pls2 = models.IntegerField()
	rsp1 = models.IntegerField()
	rsp2 = models.IntegerField()
	wtPre = models.IntegerField()
	wtPost = models.IntegerField()
