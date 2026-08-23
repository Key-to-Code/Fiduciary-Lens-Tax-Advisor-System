import os
import json
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# 1. Define synthetic profiles
dummy_profiles = [
    {
        "name": "Arjun Sharma",
        "age": 30,
        "designation": "Software Engineer",
        "location": "Bengaluru",
        "pan": "ABCDE1234F",
        "salary": 1200000,
        "hra": 240000,
        "ppf": 150000
    },
    {
        "name": "Priya Patel",
        "age": 42,
        "designation": "Product Manager",
        "location": "Mumbai",
        "pan": "WXYZ6789G",
        "salary": 2200000,
        "hra": 360000,
        "ppf": 150000
    }
]

def generate_synthetic_pdf(profile, output_path):
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    story = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle', 
        parent=styles['Heading1'], 
        fontSize=18, 
        spaceAfter=20
    )
    
    # Header Section
    story.append(Paragraph("SYNTHETIC FORM 16 - CERTIFICATE OF TAX DEDUCTED AT SOURCE", title_style))
    story.append(Spacer(1, 12))
    
    # Employee Details Table Data
    data = [
        [Paragraph("<b>Employee Name:</b>", styles['Normal']), Paragraph(profile['name'], styles['Normal']),
         Paragraph("<b>PAN:</b>", styles['Normal']), Paragraph(profile['pan'], styles['Normal'])],
        [Paragraph("<b>Age:</b>", styles['Normal']), Paragraph(str(profile['age']), styles['Normal']),
         Paragraph("<b>Designation:</b>", styles['Normal']), Paragraph(profile['designation'], styles['Normal'])],
        [Paragraph("<b>Location:</b>", styles['Normal']), Paragraph(profile['location'], styles['Normal']),
         Paragraph("<b>Assessment Year:</b>", styles['Normal']), Paragraph("2026-27", styles['Normal'])]
    ]
    
    t1 = Table(data, colWidths=[100, 150, 100, 150])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 1, colors.grey),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t1)
    story.append(Spacer(1, 20))
    
    # Financial Breakdown Table Data
    story.append(Paragraph("<b>Details of Salary Paid and Deductions under Section 80C</b>", styles['Heading2']))
    story.append(Spacer(1, 8))
    
    fin_data = [
        [Paragraph("<b>Description</b>", styles['Normal']), Paragraph("<b>Amount (₹)</b>", styles['Normal'])],
        [Paragraph("Gross Salary", styles['Normal']), Paragraph(f"{profile['salary']:,}", styles['Normal'])],
        [Paragraph("House Rent Allowance (HRA) Claimed", styles['Normal']), Paragraph(f"{profile['hra']:,}", styles['Normal'])],
        [Paragraph("Deductions under 80C (PPF/EPF)", styles['Normal']), Paragraph(f"{profile['ppf']:,}", styles['Normal'])],
        [Paragraph("Taxable Income (Standard Estimate)", styles['Normal']), Paragraph(f"{profile['salary'] - profile['hra'] - profile['ppf']:,}", styles['Normal'])]
    ]
    
    t2 = Table(fin_data, colWidths=[300, 200])
    t2.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 1, colors.grey),
        ('BACKGROUND', (0,0), (1,0), colors.whitesmoke),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t2)
    
    # Build the document
    doc.build(story)

def create_dataset(output_dir="synthetic_forms"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for i, profile in enumerate(dummy_profiles, 1):
        filename = f"Form16_Synthetic_{profile['name'].replace(' ', '_')}.pdf"
        full_path = os.path.join(output_dir, filename)
        generate_synthetic_pdf(profile, full_path)
        print(f"Generated synthetic document ({i}/{len(dummy_profiles)}): {full_path}")

if __name__ == "__main__":
    create_dataset()