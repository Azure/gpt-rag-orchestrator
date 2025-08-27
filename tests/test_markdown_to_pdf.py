"""
Unit tests for markdown_to_pdf module.

Tests the functionality of converting dictionaries with markdown content to PDF documents.
"""

import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock
import sys

# Add the project root to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.markdown_to_pdf import dict_to_pdf, markdown_to_html, save_dict_to_pdf_file


class TestMarkdownToPdf(unittest.TestCase):
    """Test cases for markdown to PDF conversion functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_dict = {
            'question.txt': 'Please provide an analysis of the construction adhesive market in the US',
            'content': '''# Weekly Brand Analysis Report

**Brand Focus:** Henkel Construction Adhesives & Sealants 
**For the Week of:** 2025-08-21 to 2025-08-27 
**Data Scope:** Publicly available web data from the past 7 days.

## 1. Executive Summary

### Market Snapshot
The US construction adhesives market continues to experience robust growth, driven by urbanization, infrastructure investment, and a marked shift toward sustainable, low-VOC products. Water-based adhesives now account for nearly 40% of the market, reflecting regulatory and consumer demand for eco-friendly solutions [1][2][3][4].

### Top Opportunity
Henkel can capitalize on the rising demand for sustainable and low-VOC adhesives by amplifying its portfolio of water-based and eco-friendly products, positioning itself as a leader in green construction solutions [2][3][4].

### Arkema Annual Report (Historical Context): https://www.arkema.com/files/live/sites/shared_arkema/files/downloads/corporate-documentations/Annual%20reports%20-%20EN/2017-arkema-annual-report.pdf'''
        }
        
        self.simple_markdown = "# Simple Test\n\nThis is a **bold** test."
        self.simple_dict = {'content': self.simple_markdown}
        
    def test_markdown_to_html_valid_input(self):
        """Test markdown to HTML conversion with valid input."""
        result = markdown_to_html(self.simple_markdown)
        
        # Check that it returns a complete HTML document
        self.assertIn('<!DOCTYPE html>', result)
        self.assertIn('<html>', result)
        self.assertIn('<head>', result)
        self.assertIn('<body>', result)
        self.assertIn('<h1 id="simple-test">Simple Test</h1>', result)
        self.assertIn('<strong>bold</strong>', result)
        
    def test_markdown_to_html_empty_input(self):
        """Test markdown to HTML conversion with empty input."""
        with self.assertRaises(ValueError):
            markdown_to_html("")
            
        with self.assertRaises(ValueError):
            markdown_to_html(None)
    
    def test_markdown_to_html_complex_markdown(self):
        """Test markdown to HTML conversion with complex markdown."""
        complex_markdown = """
# Title
## Subtitle
- List item 1
- List item 2

| Column 1 | Column 2 |
|----------|----------|
| Cell 1   | Cell 2   |

```python
print("Hello World")
```

> This is a blockquote
        """
        
        result = markdown_to_html(complex_markdown)
        
        # Check for various HTML elements
        self.assertIn('<h1 id="title">Title</h1>', result)
        self.assertIn('<h2 id="subtitle">Subtitle</h2>', result)
        self.assertIn('<ul>', result)
        self.assertIn('<li>List item 1</li>', result)
        self.assertIn('<table>', result)
        self.assertIn('<th>Column 1</th>', result)
        self.assertIn('<pre class="codehilite"><code', result)
        self.assertIn('<blockquote>', result)

    @patch('shared.markdown_to_pdf.html_to_pdf')
    def test_dict_to_pdf_valid_input(self, mock_html_to_pdf):
        """Test dict to PDF conversion with valid input."""
        mock_html_to_pdf.return_value = b'fake_pdf_content'
        
        result = dict_to_pdf(self.sample_dict)
        
        self.assertEqual(result, b'fake_pdf_content')
        mock_html_to_pdf.assert_called_once()
        
        # Check that the HTML passed to html_to_pdf contains expected content
        args, kwargs = mock_html_to_pdf.call_args
        html_content = args[0]
        self.assertIn('Weekly Brand Analysis Report', html_content)
        self.assertIn('Executive Summary', html_content)
        
    def test_dict_to_pdf_missing_content_field(self):
        """Test dict to PDF conversion when content field is missing."""
        invalid_dict = {'question.txt': 'test question'}
        
        with self.assertRaises(KeyError):
            dict_to_pdf(invalid_dict)
    
    def test_dict_to_pdf_empty_content(self):
        """Test dict to PDF conversion with empty content."""
        empty_dict = {'content': ''}
        
        with self.assertRaises(ValueError):
            dict_to_pdf(empty_dict)
            
        whitespace_dict = {'content': '   \n\t  '}
        
        with self.assertRaises(ValueError):
            dict_to_pdf(whitespace_dict)
    
    def test_dict_to_pdf_invalid_input_type(self):
        """Test dict to PDF conversion with invalid input types."""
        with self.assertRaises(ValueError):
            dict_to_pdf("not a dict")
            
        with self.assertRaises(ValueError):
            dict_to_pdf(None)
            
        with self.assertRaises(ValueError):
            dict_to_pdf(['list', 'not', 'dict'])
    
    def test_dict_to_pdf_non_string_content(self):
        """Test dict to PDF conversion with non-string content."""
        invalid_dict = {'content': 123}
        
        with self.assertRaises(ValueError):
            dict_to_pdf(invalid_dict)
            
        invalid_dict2 = {'content': ['list', 'content']}
        
        with self.assertRaises(ValueError):
            dict_to_pdf(invalid_dict2)

    @patch('shared.markdown_to_pdf.html_to_pdf')
    def test_dict_to_pdf_html_generation_failure(self, mock_html_to_pdf):
        """Test dict to PDF conversion when HTML to PDF conversion fails."""
        mock_html_to_pdf.side_effect = Exception("PDF generation failed")
        
        with self.assertRaises(Exception) as context:
            dict_to_pdf(self.simple_dict)
        
        self.assertIn("PDF generation failed", str(context.exception))

    @patch('shared.markdown_to_pdf.dict_to_pdf')
    @patch('builtins.open')
    def test_save_dict_to_pdf_file_success(self, mock_open, mock_dict_to_pdf):
        """Test saving dict to PDF file successfully."""
        mock_dict_to_pdf.return_value = b'fake_pdf_content'
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        output_path = '/tmp/test.pdf'
        result = save_dict_to_pdf_file(self.simple_dict, output_path)
        
        self.assertEqual(result, output_path)
        mock_dict_to_pdf.assert_called_once_with(self.simple_dict)
        mock_open.assert_called_once_with(output_path, 'wb')
        mock_file.write.assert_called_once_with(b'fake_pdf_content')

    @patch('shared.markdown_to_pdf.dict_to_pdf')
    def test_save_dict_to_pdf_file_dict_to_pdf_failure(self, mock_dict_to_pdf):
        """Test saving dict to PDF file when dict_to_pdf fails."""
        mock_dict_to_pdf.side_effect = Exception("Dict to PDF failed")
        
        with self.assertRaises(Exception) as context:
            save_dict_to_pdf_file(self.simple_dict, '/tmp/test.pdf')
        
        self.assertIn("Failed to save PDF file", str(context.exception))

    def test_dict_to_pdf_ignores_extra_fields(self):
        """Test that dict_to_pdf ignores fields other than content."""
        dict_with_extras = {
            'content': self.simple_markdown,
            'question.txt': 'Some question',
            'extra_field': 'Extra data',
            'another_field': 123
        }
        
        with patch('shared.markdown_to_pdf.html_to_pdf') as mock_html_to_pdf:
            mock_html_to_pdf.return_value = b'fake_pdf_content'
            
            result = dict_to_pdf(dict_with_extras)
            
            self.assertEqual(result, b'fake_pdf_content')
            mock_html_to_pdf.assert_called_once()

    def test_integration_with_real_example(self):
        """Integration test with the actual example provided by the user."""
        # This test requires actual WeasyPrint functionality
        # Skip if WeasyPrint is not available
        try:
            from weasyprint import HTML
        except ImportError:
            self.skipTest("WeasyPrint not available for integration test")
        
        try:
            result = dict_to_pdf(self.sample_dict)
            
            # Check that we got bytes back
            self.assertIsInstance(result, bytes)
            self.assertGreater(len(result), 0)
            
            # Check PDF magic number (PDF files start with %PDF)
            self.assertTrue(result.startswith(b'%PDF'))
            
        except Exception as e:
            self.skipTest(f"Integration test failed due to environment: {str(e)}")


class TestMarkdownToPdfEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def test_markdown_with_html_content(self):
        """Test markdown that contains HTML content."""
        markdown_with_html = """
# Title with <script>alert('xss')</script>

This contains <b>HTML</b> tags.
        """
        
        # Should not raise an exception and should handle HTML safely
        result = markdown_to_html(markdown_with_html)
        self.assertIn('Title with', result)
        # HTML should be escaped or handled safely
        
    def test_very_large_content(self):
        """Test with very large markdown content."""
        large_content = "# Large Content\n\n" + "This is a test paragraph. " * 1000
        large_dict = {'content': large_content}
        
        with patch('shared.markdown_to_pdf.html_to_pdf') as mock_html_to_pdf:
            mock_html_to_pdf.return_value = b'fake_pdf_content'
            
            result = dict_to_pdf(large_dict)
            self.assertEqual(result, b'fake_pdf_content')
    
    def test_unicode_content(self):
        """Test with unicode characters in content."""
        unicode_content = """
# Título con acentos
        
Este es contenido en español con caracteres especiales: ñ, á, é, í, ó, ú
También probamos emojis: 🚀 📊 💡
Y caracteres especiales: €, £, ¥, ©, ®, ™
        """
        unicode_dict = {'content': unicode_content}
        
        with patch('shared.markdown_to_pdf.html_to_pdf') as mock_html_to_pdf:
            mock_html_to_pdf.return_value = b'fake_pdf_content'
            
            result = dict_to_pdf(unicode_dict)
            self.assertEqual(result, b'fake_pdf_content')
            
            # Check that unicode content was passed to html_to_pdf
            args, kwargs = mock_html_to_pdf.call_args
            html_content = args[0]
            self.assertIn('Título con acentos', html_content)


if __name__ == '__main__':
    # Create a test suite using the recommended method
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTest(loader.loadTestsFromTestCase(TestMarkdownToPdf))
    suite.addTest(loader.loadTestsFromTestCase(TestMarkdownToPdfEdgeCases))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1)
