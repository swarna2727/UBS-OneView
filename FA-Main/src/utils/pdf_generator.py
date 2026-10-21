"""PDF generation utilities for investment memos."""

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime
import os
import re
from typing import Dict, Any

class PDFGenerator:
    """Generate professional PDF reports from investment memo data."""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Set up custom paragraph styles for professional investment reports."""
        # Helper function to safely add styles
        def add_style_if_not_exists(name, style):
            if name not in self.styles:
                self.styles.add(style)
        
        # Main title style
        add_style_if_not_exists('CustomTitle', ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=24,
            spaceBefore=12,
            textColor=colors.darkblue,
            alignment=1,  # Center alignment
            fontName='Helvetica-Bold'
        ))
        
        # Section headers
        add_style_if_not_exists('SectionHeader', ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.darkblue,
            fontName='Helvetica-Bold',
            borderWidth=1,
            borderColor=colors.lightgrey,
            borderPadding=8,
            backColor=colors.lightgrey
        ))
        
        # Subsection headers
        add_style_if_not_exists('SubsectionHeader', ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceAfter=8,
            spaceBefore=12,
            textColor=colors.darkblue,
            fontName='Helvetica-Bold'
        ))
        
        # Body text
        add_style_if_not_exists('BodyText', ParagraphStyle(
            name='BodyText',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=6,
            spaceBefore=0,
            fontName='Helvetica',
            leading=14,
            alignment=0  # Left alignment
        ))
        
        # Bullet points
        add_style_if_not_exists('BulletPoint', ParagraphStyle(
            name='BulletPoint',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=4,
            spaceBefore=2,
            fontName='Helvetica',
            leftIndent=20,
            leading=12
        ))
        
        # Risk levels
        add_style_if_not_exists('RiskHigh', ParagraphStyle(
            name='RiskHigh',
            parent=self.styles['Normal'],
            textColor=colors.red,
            fontSize=11,
            fontName='Helvetica-Bold',
            leftIndent=20
        ))
        
        add_style_if_not_exists('RiskMedium', ParagraphStyle(
            name='RiskMedium',
            parent=self.styles['Normal'],
            textColor=colors.orange,
            fontSize=11,
            fontName='Helvetica-Bold',
            leftIndent=20
        ))
        
        add_style_if_not_exists('RiskLow', ParagraphStyle(
            name='RiskLow',
            parent=self.styles['Normal'],
            textColor=colors.green,
            fontSize=11,
            fontName='Helvetica-Bold',
            leftIndent=20
        ))
        
        # Executive summary style
        add_style_if_not_exists('ExecutiveSummary', ParagraphStyle(
            name='ExecutiveSummary',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=8,
            spaceBefore=8,
            fontName='Helvetica',
            leading=16,
            backColor=colors.lightgrey,
            borderWidth=1,
            borderColor=colors.grey,
            borderPadding=10
        ))
        
        # Financial projections style
        add_style_if_not_exists('FinancialText', ParagraphStyle(
            name='FinancialText',
            parent=self.styles['Normal'],
            fontSize=11,
            fontName='Helvetica',
            textColor=colors.darkgreen,
            leading=13
        ))
        
        # Emphasis style
        add_style_if_not_exists('EmphasisText', ParagraphStyle(
            name='EmphasisText',
            parent=self.styles['Normal'],
            fontSize=11,
            fontName='Helvetica-Bold',
            leading=13
        ))
    
    def generate_investment_memo_pdf(self, memo_data: Dict[Any, Any], output_path: str) -> str:
        """
        Generate a professional investment memo PDF.
        
        Args:
            memo_data: Dictionary containing memo data from Agent 5
            output_path: Path where PDF will be saved
            
        Returns:
            Path to generated PDF file
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            story = []
            
            # Title
            title = Paragraph("Investment Strategy Memo", self.styles['CustomTitle'])
            story.append(title)
            story.append(Spacer(1, 12))
            
            # Document info table
            metadata = memo_data.get('metadata', {})
            doc_info = [
                ['Transaction ID:', metadata.get('transaction_id', 'N/A')],
                ['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                ['Processing Time:', f"{metadata.get('total_processing_time_ms', 0)} ms"]
            ]
            
            info_table = Table(doc_info, colWidths=[2*inch, 4*inch])
            info_table.setStyle(TableStyle([
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ]))
            
            story.append(info_table)
            story.append(Spacer(1, 24))
            
            # Extract report content
            agent_5_output = memo_data.get('agent_5_output', {})
            report_text = agent_5_output.get('report_text', '')
            
            if report_text:
                # Parse the formatted text report and convert to PDF sections
                self._add_formatted_text_to_story(story, report_text)
            else:
                # Fallback to old format if no text report
                memo_content = memo_data.get('investment_memo', {})
                self._add_legacy_content_to_story(story, memo_content)
            
            # Build PDF
            doc.build(story)
            return output_path
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"PDF generation failed: {str(e)}")
            
            # Create a simple fallback PDF
            return self._create_fallback_pdf(memo_data, output_path)
    
    def _clean_markdown_formatting(self, text: str) -> str:
        """Remove markdown formatting and clean up text for PDF."""
        
        # Remove markdown bold formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        
        # Remove markdown italic formatting
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        
        # Clean up multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Clean up multiple newlines
        text = re.sub(r'\n\n\n+', '\n\n', text)
        
        # Remove email-style headers and proposal metadata
        lines = text.split('\n')
        filtered_lines = []
        
        for line in lines:
            line_stripped = line.strip()
            # Skip email header patterns and proposal metadata
            if (line_stripped.startswith('TO:') or 
                line_stripped.startswith('FROM:') or 
                line_stripped.startswith('DATE:') or 
                line_stripped.startswith('SUBJECT:') or
                line_stripped.startswith('**Strategy Name:**') or
                line_stripped.startswith('**Proposal Date:**') or
                line_stripped.startswith('**Prepared By:**') or
                line_stripped.startswith('**Decision Required:**')):
                continue
            else:
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    def _add_formatted_text_to_story(self, story, report_text):
        """Add formatted text report to PDF story with enhanced formatting."""
        
        # Clean markdown formatting first
        report_text = self._clean_markdown_formatting(report_text)
        
        lines = report_text.split('\n')
        current_section = None
        in_bullet_list = False
        table_data = []
        in_table = False

        for line in lines:
            line = line.strip()
            if not line:
                if in_bullet_list:
                    story.append(Spacer(1, 4))
                    in_bullet_list = False
                if in_table and table_data:
                    self._add_table_to_story(story, table_data)
                    table_data = []
                    in_table = False
                continue

            # Check for main section headers (# Header)
            if line.startswith('# '):
                # Finish any open table
                if in_table and table_data:
                    self._add_table_to_story(story, table_data)
                    table_data = []
                    in_table = False
                
                section_title = line[2:].strip()
                if current_section:
                    story.append(Spacer(1, 16))
                story.append(Paragraph(section_title, self.styles['SectionHeader']))
                current_section = section_title
                in_bullet_list = False
                
            # Check for subsection headers (## Subheader)
            elif line.startswith('## '):
                # Finish any open table
                if in_table and table_data:
                    self._add_table_to_story(story, table_data)
                    table_data = []
                    in_table = False
                
                subsection_title = line[3:].strip()
                story.append(Spacer(1, 8))
                story.append(Paragraph(subsection_title, self.styles['SubsectionHeader']))
                in_bullet_list = False
                
            # Check for table rows (containing |)
            elif '|' in line and line.count('|') >= 2:
                if not in_table:
                    in_table = True
                    table_data = []
                
                # Parse table row
                cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                if cells:  # Only add non-empty rows
                    table_data.append(cells)
                in_bullet_list = False
                
            # Check for bullet points (- or * )
            elif line.startswith('- ') or line.startswith('* '):
                # Finish any open table
                if in_table and table_data:
                    self._add_table_to_story(story, table_data)
                    table_data = []
                    in_table = False
                
                bullet_text = line[2:].strip()
                # Clean up bullet text
                if bullet_text.startswith('• '):
                    bullet_text = bullet_text[2:].strip()
                
                # Apply special formatting for risk levels and financial data
                if 'High' in bullet_text and ('risk' in bullet_text.lower() or 'probability' in bullet_text.lower()):
                    story.append(Paragraph(f"• {bullet_text}", self.styles['RiskHigh']))
                elif 'Medium' in bullet_text and ('risk' in bullet_text.lower() or 'probability' in bullet_text.lower()):
                    story.append(Paragraph(f"• {bullet_text}", self.styles['RiskMedium']))
                elif 'Low' in bullet_text and ('risk' in bullet_text.lower() or 'probability' in bullet_text.lower()):
                    story.append(Paragraph(f"• {bullet_text}", self.styles['RiskLow']))
                elif any(term in bullet_text for term in ['%', '$', 'return', 'ratio', 'margin']):
                    story.append(Paragraph(f"• {bullet_text}", self.styles['FinancialText']))
                else:
                    story.append(Paragraph(f"• {bullet_text}", self.styles['BulletPoint']))
                in_bullet_list = True
                
            # Check for numbered lists
            elif line[0].isdigit() and '. ' in line[:5]:
                # Finish any open table
                if in_table and table_data:
                    self._add_table_to_story(story, table_data)
                    table_data = []
                    in_table = False
                
                # Extract number and text
                parts = line.split('. ', 1)
                if len(parts) == 2:
                    number = parts[0]
                    text = parts[1]
                    story.append(Paragraph(f"{number}. {text}", self.styles['BulletPoint']))
                    in_bullet_list = True
                else:
                    story.append(Paragraph(line, self.styles['BodyText']))
                    in_bullet_list = False
                    
            else:
                # Finish any open table
                if in_table and table_data:
                    self._add_table_to_story(story, table_data)
                    table_data = []
                    in_table = False
                
                # Regular paragraph
                if line:
                    # Special formatting based on content and section
                    if current_section and 'executive summary' in current_section.lower():
                        story.append(Paragraph(line, self.styles['ExecutiveSummary']))
                    elif any(term in line for term in ['%', '$', 'return', 'ratio', 'volatility', 'margin']):
                        story.append(Paragraph(line, self.styles['FinancialText']))
                    elif any(keyword in line.lower() for keyword in ['recommend', 'approve', 'reject', 'critical', 'important']):
                        story.append(Paragraph(line, self.styles['EmphasisText']))
                    else:
                        story.append(Paragraph(line, self.styles['BodyText']))
                    in_bullet_list = False

        # Add any remaining table
        if in_table and table_data:
            self._add_table_to_story(story, table_data)

        story.append(Spacer(1, 16))
    
    def _add_table_to_story(self, story, table_data):
        """Add a properly formatted table to the PDF story."""
        if not table_data:
            return
        
        # Clean table data
        cleaned_data = []
        for row in table_data:
            if len(row) > 1:  # Only include rows with multiple columns
                cleaned_data.append(row)
        
        if not cleaned_data:
            return
        
        # Calculate column widths based on content
        max_cols = max(len(row) for row in cleaned_data)
        col_width = 6 * inch / max_cols  # Distribute evenly
        
        table = Table(cleaned_data, colWidths=[col_width] * max_cols)
        
        # Style the table
        table_style = [
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ]
        
        # Header row styling if it looks like a header
        if cleaned_data and any(word in str(cleaned_data[0]).lower() for word in ['scenario', 'year', 'metric', 'category']):
            table_style.extend([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ])
        
        table.setStyle(TableStyle(table_style))
        story.append(table)
        story.append(Spacer(1, 8))
    
    def _create_fallback_pdf(self, memo_data: Dict[Any, Any], output_path: str) -> str:
        """Create a simple fallback PDF when main generation fails."""
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            story = []
            
            # Simple title
            title = Paragraph("Investment Strategy Memo", self.styles['CustomTitle'])
            story.append(title)
            story.append(Spacer(1, 12))
            
            # Error message
            error_msg = Paragraph(
                "PDF generation encountered formatting issues. The memo content has been simplified for display. Please refer to the JSON output for complete technical details.",
                self.styles['BodyText']
            )
            story.append(error_msg)
            story.append(Spacer(1, 12))
            
            # Basic info
            metadata = memo_data.get('metadata', {})
            basic_info = [
                ['Transaction ID:', metadata.get('transaction_id', 'N/A')],
                ['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                ['Status:', 'Simplified PDF - See JSON for full details']
            ]
            
            info_table = Table(basic_info, colWidths=[2*inch, 4*inch])
            info_table.setStyle(TableStyle([
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ]))
            
            story.append(info_table)
            story.append(Spacer(1, 16))
            
            # Try to extract basic content
            agent_5_output = memo_data.get('agent_5_output', {})
            report_text = agent_5_output.get('report_text', '')
            
            if report_text:
                # Add simplified version of the text
                simplified_text = self._clean_markdown_formatting(report_text)
                # Split into paragraphs and add basic formatting
                paragraphs = simplified_text.split('\n\n')
                for para in paragraphs[:10]:  # Limit to first 10 paragraphs
                    if para.strip():
                        if para.strip().startswith('#'):
                            # Header
                            header_text = para.strip().lstrip('#').strip()
                            story.append(Paragraph(header_text, self.styles['SectionHeader']))
                        else:
                            # Regular paragraph
                            story.append(Paragraph(para.strip(), self.styles['BodyText']))
                        story.append(Spacer(1, 6))
            
            # Build PDF
            doc.build(story)
            return output_path
            
        except Exception as fallback_error:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Fallback PDF creation also failed: {str(fallback_error)}")
            # Return the original path even if fallback fails
            return output_path
    
    def _add_legacy_content_to_story(self, story, memo_content):
        """Add legacy JSON-based content to PDF story (fallback)."""
        # Executive Summary
        story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        story.append(Paragraph(memo_content.get('executive_summary', 'No executive summary available.'), self.styles['ExecutiveSummary']))
        story.append(Spacer(1, 12))
        
        # Strategy Overview
        story.append(Paragraph("Strategy Overview", self.styles['SectionHeader']))
        story.append(Paragraph(memo_content.get('strategy_overview', 'No strategy overview available.'), self.styles['BodyText']))
        story.append(Spacer(1, 12))
        
        # Key Risks
        story.append(Paragraph("Key Risks", self.styles['SectionHeader']))
        
        key_risks = memo_content.get('key_risks', [])
        if key_risks:
            risk_data = [['Risk', 'Severity', 'Mitigation']]
            
            for i, risk in enumerate(key_risks[:3], 1):
                severity = risk.get('severity', 'Medium')
                risk_data.append([
                    f"{i}. {risk.get('risk', 'Risk not specified')}",
                    severity,
                    risk.get('mitigation', 'Mitigation not specified')
                ])
            
            risk_table = Table(risk_data, colWidths=[2.5*inch, 1*inch, 2.5*inch])
            risk_table.setStyle(TableStyle([
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ]))
            
            story.append(risk_table)
        else:
            story.append(Paragraph("No risk information available.", self.styles['BodyText']))
        
        story.append(Spacer(1, 12))
        
        # Approval Reasons
        story.append(Paragraph("Reasons for Approval", self.styles['SectionHeader']))
        
        approval_reasons = memo_content.get('approval_reasons', [])
        if approval_reasons:
            for i, reason in enumerate(approval_reasons[:3], 1):
                story.append(Paragraph(f"{i}. {reason.get('reason', 'Reason not specified')}", self.styles['BodyText']))
                story.append(Paragraph(f"Supporting Data: {reason.get('supporting_data', 'No supporting data provided')}", self.styles['BodyText']))
                story.append(Spacer(1, 6))
        else:
            story.append(Paragraph("No approval reasons available.", self.styles['BodyText']))
        
        # Implementation Timeline
        story.append(Paragraph("Implementation Timeline", self.styles['SectionHeader']))
        story.append(Paragraph(memo_content.get('implementation_timeline', 'No implementation timeline provided.'), self.styles['BodyText']))
        story.append(Spacer(1, 12))
        
        # Resource Requirements
        story.append(Paragraph("Resource Requirements", self.styles['SectionHeader']))
        
        resources = memo_content.get('resource_requirements', {})
        if resources:
            resource_data = [
                ['Capital:', resources.get('capital', 'Not specified')],
                ['Personnel:', resources.get('personnel', 'Not specified')],
                ['Technology:', resources.get('technology', 'Not specified')],
                ['Data Feeds:', ', '.join(resources.get('data_feeds', ['Not specified']))]
            ]
            
            resource_table = Table(resource_data, colWidths=[1.5*inch, 4.5*inch])
            resource_table.setStyle(TableStyle([
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ]))
            
            story.append(resource_table)
        else:
            story.append(Paragraph("No resource requirements available.", self.styles['BodyText']))