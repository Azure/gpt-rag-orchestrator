"""
Markdown to PDF conversion utility.

This module provides functionality to convert markdown content to PDF documents
using xhtml2pdf for PDF generation without native dependencies.
"""

import io
import logging
import markdown
from typing import Dict, Any
from xhtml2pdf import pisa

# Set up logging
logger = logging.getLogger(__name__)


def break_long_urls(text: str, max_length: int = 130) -> str:
    """
    Break long URLs by inserting line breaks at strategic points.
    Uses adaptive break percentages based on URL length:
    the adaptative break generates a percentage based on the URL length and the max_length
    getting the inverse of the percentage gives us the break point

    Args:
        text (str): Text content that may contain long URLs
        max_length (int): Maximum URL length before breaking (default: 130)

    Returns:
        str: Text with long URLs broken into multiple lines
    """
    import re

    # Pattern to match URLs (http/https, ftp, and www)
    url_pattern = r'(https?://[^\s<>"{}|\\^`\[\]]+|ftp://[^\s<>"{}|\\^`\[\]]+|www\.[^\s<>"{}|\\^`\[\]]+)'

    def break_url(match):
        url = match.group(1)

        # Only break URLs longer than max_length
        if len(url) <= max_length:
            return url

        # Get adaptive break percentage based on URL length
        adaptive_percent = 1/(len(url)/ max_length)
        break_point = int(len(url) * adaptive_percent)

        # Log the adaptive breaking decision
        logger.debug(
            f"Breaking URL of length {len(url)} at {adaptive_percent*100:.0f}% (position {break_point})")

        # Try to break at a natural point (after / or ? or &)
        natural_breaks = ['/', '?', '&', '=', '-', '_']
        best_break = break_point

        # Look for natural break points within 10 characters of the calculated break point
        for i in range(max(0, break_point - 10), min(len(url), break_point + 10)):
            if url[i] in natural_breaks:
                best_break = i + 1  # Break after the character
                break

        # Insert line break
        broken_url = url[:best_break] + '\n' + url[best_break:]

        # If the remaining part is still too long, break it again
        remaining = url[best_break:]
        if len(remaining) > max_length:
            # For the second break, just break at max_length
            second_break = max_length
            broken_url = url[:best_break] + '\n' + \
                remaining[:second_break] + '\n' + remaining[second_break:]

        return broken_url

    # Apply URL breaking to all URLs in the text
    return re.sub(url_pattern, break_url, text)


def html_to_pdf_xhtml2pdf(html_content: str) -> bytes:
    """
    Convert HTML content to PDF bytes using xhtml2pdf.

    Args:
        html_content (str): The HTML content to convert

    Returns:
        bytes: PDF document as bytes

    Raises:
        Exception: If PDF generation fails
    """
    try:
        # Create a bytes buffer for the PDF output
        pdf_buffer = io.BytesIO()

        # Convert HTML to PDF using xhtml2pdf
        pisa_status = pisa.CreatePDF(html_content, dest=pdf_buffer)

        if pisa_status.err:
            raise Exception(
                f"xhtml2pdf conversion failed with {pisa_status.err} errors")

        # Get the PDF bytes
        pdf_bytes = pdf_buffer.getvalue()
        pdf_buffer.close()

        return pdf_bytes

    except Exception as e:
        logger.error(
            f"Failed to convert HTML to PDF using xhtml2pdf: {str(e)}")
        raise Exception(f"HTML to PDF conversion failed: {str(e)}") from e


def markdown_to_html(markdown_content: str) -> str:
    """
    Convert markdown content to HTML with proper formatting.

    Args:
        markdown_content (str): The markdown content to convert

    Returns:
        str: HTML formatted content

    Raises:
        ValueError: If markdown_content is empty or None
    """
    if not markdown_content:
        raise ValueError("Markdown content cannot be empty or None")

    # Pre-process the markdown content to break long URLs with adaptive breaking
    processed_content = break_long_urls(markdown_content, max_length=130)

    # Use markdown library with extensions for better formatting
    md = markdown.Markdown(extensions=[
        'fenced_code',
        'tables',
        'toc',
        'nl2br',  # Convert newlines to <br>
        'sane_lists',
        'codehilite'  # Code highlighting
    ])

    html_content = md.convert(processed_content)

    # Wrap in a complete HTML document with basic styling
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Generated Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 40px;
                color: #333;
                word-wrap: break-word;
                width: 100%;
            }}
            h1, h2, h3, h4, h5, h6 {{
                color: #2c3e50;
                margin-top: 30px;
                margin-bottom: 15px;
                word-wrap: break-word;
                white-space: normal;
            }}
            h1 {{
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                border-bottom: 1px solid #bdc3c7;
                padding-bottom: 5px;
            }}
            p {{
                margin-bottom: 15px;
                word-wrap: break-word;
                white-space: normal;
            }}
            ul, ol {{
                margin-bottom: 15px;
                margin-top: 10px;
                padding-left: 25px;
            }}
            li {{
                margin-bottom: 3px;
                margin-top: 2px;
                line-height: 1.4;
                word-wrap: break-word;
                white-space: normal;
            }}
            /* Reduce spacing for nested lists */
            ul ul, ol ol, ul ol, ol ul {{
                margin-top: 5px;
                margin-bottom: 5px;
                padding-left: 20px;
            }}
            /* First and last list items */
            li:first-child {{
                margin-top: 0;
            }}
            li:last-child {{
                margin-bottom: 0;
            }}
            /* Nested list items */
            li li {{
                margin-bottom: 2px;
                margin-top: 1px;
            }}
            /* Remove extra spacing from paragraphs inside list items */
            li p {{
                margin-bottom: 5px;
                margin-top: 0;
            }}
            code {{
                background-color: #f8f9fa;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
                word-wrap: break-word;
                white-space: normal;
                font-size: 90%;
            }}
            pre {{
                background-color: #f8f9fa;
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 15px;
                white-space: pre-wrap;
                word-wrap: break-word;
                font-size: 85%;
                font-family: 'Courier New', monospace;
                width: 100%;
            }}
            blockquote {{
                border-left: 4px solid #3498db;
                margin: 20px 0;
                padding-left: 20px;
                color: #666;
                word-wrap: break-word;
                white-space: normal;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
                word-wrap: break-word;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
                word-wrap: break-word;
                white-space: normal;
                vertical-align: top;
                width: 33%;
            }}
            th {{
                background-color: #f2f2f2;
                font-weight: bold;
            }}
            a {{
                color: #3498db;
                text-decoration: none;
                word-wrap: break-word;
                white-space: normal;
            }}
            a:hover {{
                text-decoration: underline;
            }}
            /* Images */
            img {{
                width: 100%;
                height: auto;
            }}
            /* Divs and spans */
            div, span {{
                word-wrap: break-word;
                white-space: normal;
            }}
            /* Force break very long words */
            p, li, td, th, blockquote, div {{
                word-break: break-word;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    return full_html


def dict_to_pdf(input_dict: Dict[str, Any]) -> bytes:
    """
    Convert a dictionary containing markdown content to a PDF document.

    Expected dictionary format:
    {
        'content': 'markdown content here',
        'question.txt': 'optional question or description',  # ignored
        # other fields are ignored
    }

    Args:
        input_dict (Dict[str, Any]): Dictionary containing 'content' field with markdown

    Returns:
        bytes: PDF document as bytes

    Raises:
        ValueError: If input_dict is invalid or missing required fields
        KeyError: If 'content' field is missing
        Exception: If PDF generation fails
    """

    if 'content' not in input_dict:
        raise KeyError("Dictionary must contain a 'content' field")

    markdown_content = input_dict['content']

    if not isinstance(markdown_content, str):
        raise ValueError("Content field must be a string")

    if not markdown_content.strip():
        raise ValueError("Content field cannot be empty")

    try:
        logger.info(
            f"Converting markdown content to PDF (length: {len(markdown_content)} chars)")

        # Convert markdown to HTML
        html_content = markdown_to_html(markdown_content)

        # Convert HTML to PDF using xhtml2pdf
        pdf_bytes = html_to_pdf_xhtml2pdf(html_content)

        logger.info(f"Successfully generated PDF ({len(pdf_bytes)} bytes)")
        return pdf_bytes

    except Exception as e:
        logger.error(f"Failed to convert markdown to PDF: {str(e)}")
        raise Exception(f"PDF generation failed: {str(e)}") from e


def save_dict_to_pdf_file(input_dict: Dict[str, Any], output_path: str) -> str:
    """
    Convert a dictionary containing markdown content to a PDF file.

    Args:
        input_dict (Dict[str, Any]): Dictionary containing 'content' field with markdown
        output_path (str): Path where the PDF file should be saved

    Returns:
        str: Path to the saved PDF file

    Raises:
        ValueError: If input_dict is invalid or missing required fields
        KeyError: If 'content' field is missing
        Exception: If PDF generation or file saving fails
    """
    try:
        # Generate PDF bytes
        pdf_bytes = dict_to_pdf(input_dict)

        # Write to file
        with open(output_path, 'wb') as f:
            f.write(pdf_bytes)

        logger.info(f"PDF saved to: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to save PDF to file: {str(e)}")
        raise Exception(f"Failed to save PDF file: {str(e)}") from e


if __name__ == "__main__":
    input_dict = {
        'content': '''# Comprehensive Markdown Style Test

This document contains various markdown elements to test PDF conversion styling.

## Table of Contents
1. [Headers](#headers)
2. [Text Formatting](#text-formatting)
3. [Lists](#lists)
4. [Tables](#tables)
5. [Code](#code)
6. [Links and Images](#links-and-images)
7. [Quotes](#quotes)

---

## Headers

### Level 3 Header
#### Level 4 Header
##### Level 5 Header
###### Level 6 Header

## Text Formatting

This paragraph demonstrates **bold text**, *italic text*, and ***bold italic text***. 

You can also use `inline code` within sentences. Here's some ~~strikethrough text~~ and some regular text.

> This is a blockquote. It should have a distinct visual style with a left border and different background or text color.

## Lists

### Unordered Lists
- First item
- Second item with a longer description that might wrap to multiple lines
  - Nested item 1
  - Nested item 2
    - Double nested item
- Third item

### Ordered Lists
1. First numbered item
2. Second numbered item
   1. Nested numbered item
   2. Another nested item
3. Third numbered item with **bold formatting**

### Mixed Lists
1. Start with numbered
   - Switch to bullets
   - Another bullet
2. Back to numbered
   - More bullets with `inline code`

## Tables

### Simple Table
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Row 1, Col 1 | Row 1, Col 2 | Row 1, Col 3 |
| Row 2, Col 1 | Row 2, Col 2 | Row 2, Col 3 |

### Table with Formatting
| Feature | Status | Notes |
|---------|--------|-------|
| **Bold Headers** | ✅ Complete | Working correctly |
| *Italic Text* | ⚠️ Partial | Needs testing |
| `Code Formatting` | ❌ Pending | Not implemented |
| Long text that might wrap in cells | ✅ Complete | Should handle overflow gracefully |

### Aligned Table
| Left Aligned | Center Aligned | Right Aligned |
|:-------------|:--------------:|--------------:|
| Left | Center | Right |
| This is left | This is center | This is right |

## Code

### Inline Code
Use `print("Hello World")` for basic output. Variables like `user_name` should be in code format.

### Code Blocks

```python
def hello_world():
    """A simple hello world function."""
    print("Hello, World!")
    return True

# This is a comment
user_name = "John Doe"
result = hello_world()
```

```javascript
// JavaScript example
function calculateTotal(items) {
    return items.reduce((sum, item) => sum + item.price, 0);
}

const items = [
    { name: "Item 1", price: 10.99 },
    { name: "Item 2", price: 25.50 }
];
```

```sql
-- SQL example
SELECT 
    customer_id,
    customer_name,
    COUNT(*) as order_count,
    SUM(total_amount) as total_spent
FROM orders 
WHERE order_date >= '2023-01-01'
GROUP BY customer_id, customer_name
ORDER BY total_spent DESC;
```

## Links and Images

Here are some example links:
- [External link to Google](https://www.google.com)
- [Internal link](#headers)

## Quotes

> "The best way to predict the future is to create it." - Peter Drucker

> This is a longer blockquote that spans multiple lines and should be properly formatted with appropriate margins and styling to distinguish it from regular paragraph text.

### Nested Quotes
> This is a quote
> 
> > This is a nested quote within the first quote
> > 
> > It should have different styling
> 
> Back to the original quote level

## Special Characters and Symbols

Here are some special characters: © ® ™ € $ £ ¥ § ¶ † ‡ • … ‰ ′ ″ ‴ ※ ‼ ⁇ ⁈ ⁉

Mathematical symbols: ± × ÷ ∞ ≠ ≤ ≥ ≈ ∑ ∏ √ ∂ ∫ Ω α β γ δ π

Arrows: ← → ↑ ↓ ↔ ↕ ↖ ↗ ↘ ↙

## Horizontal Rules

Above this line there should be a horizontal rule.

---

Below this line there should be another horizontal rule.

## Mixed Content Example

1. **Project Setup**
   - Install dependencies: `npm install`
   - Configure environment variables
   
2. **Database Configuration**
   
   | Environment | Host | Port |
   |-------------|------|------|
   | Development | localhost | 5432 |
   | Production | prod-db.example.com | 5432 |

3. **Code Implementation**
   
   ```python
   # Example configuration
   config = {
       "database": {
           "host": "localhost",
           "port": 5432,
           "name": "myapp"
       }
   }
   ```

4. **Long URLs and Links Test**

## 🔗 Long URLs and Links Test

### Very Long URLs
This paragraph contains a very long URL that should wrap properly: https://www.example.com/very/long/path/that/might/cause/overflow/issues/in/pdf/generation/with/many/parameters?param1=verylongvalue1&param2=anotherlongvalue2&param3=yetanotherlongvalue3&param4=extremelylongvalue4&param5=superlongvalue5&param6=incrediblylong

[1]https://www.example.com/very/long/path/that/might/cause/overflow/issues/in/pdf/generation/with/many/parameters?param1=verylongvalue1&param2=anotherlongvalue2&param3=yetanotherlongvalue3&param4=extremelylongvalue4&param5=superlongvalue5&param6=incrediblylong

[2]https://www.example.com/very/long/path/that/might/cause/overflow/issues/in/pdf/generation/with/many/parameters?param1=verylongvalue1&param2=anotherlongvalue2&param3=yetanotherlongvalue3&param4=extremelylongvalue4&param5=superlongvalue5&param6=incrediblylong

[3]https://www.example.com/very/long/path/that/might/cause/overflow/issues/in/pdf/generation/with/many/parameters?param1=verylongvalue1&param2=anotherlongvalue2&param3=yetanotherlongvalue3&param4=extremelylongvalue4&param5=superlongvalue5&param6=incrediblylong

### Multiple Long URLs in List
- First URL: https://api.example.com/v1/users/12345/documents/reports/financial/quarterly/2023/q4/detailed-analysis-with-charts-and-graphs
- Second URL: https://dashboard.analytics.company.com/reports/user-engagement/monthly/2023/december/detailed-breakdown-by-demographics-and-regions
- Third URL: https://storage.cloudprovider.com/buckets/company-data/backups/database-dumps/postgresql/production/2023-12-31-full-backup-with-indexes

## 📝 Very Long Words Test

### Scientific and Technical Terms
This paragraph contains supercalifragilisticexpialidocious and pneumonoultramicroscopicsilicovolcanoconiosiswhichisaverylongwordthatmightcauseoverflowissues in the same sentence.

### Programming-Related Long Names
- `VeryLongClassNameThatExceedsNormalLengthLimitsAndMightCauseOverflowIssuesInPDFGeneration`
- `extremely_long_variable_name_that_follows_snake_case_convention_but_is_unreasonably_long_for_demonstration_purposes`
- `AnotherExtremelyLongMethodNameThatFollowsCamelCaseButIsWayTooLongForPracticalUse()`

## 💻 Long Code Blocks Test

### Python Code with Long Lines
```python
def very_long_function_name_that_might_cause_overflow_issues_in_pdf_generation_when_displayed_in_code_blocks():
    """
    This is a very long docstring that explains what this function does in great detail.
    It spans multiple lines and contains very long sentences that should wrap properly
    without causing any overflow issues in the PDF generation process.
    """
    very_long_variable_name_that_exceeds_normal_line_length_limits = "This is a very long string that might cause overflow issues in PDF generation when displayed in code blocks and should be handled gracefully by the text wrapping system"
    
    another_extremely_long_variable_name_for_demonstration_purposes = {
        "very_long_key_name_that_might_cause_issues": "very_long_value_that_should_wrap_properly",
        "another_long_key_for_testing_purposes": "another_long_value_with_lots_of_text_content",
        "url_key": "https://www.example.com/very/long/url/that/might/cause/overflow/issues/in/code/blocks"
    }
    
    return very_long_variable_name_that_exceeds_normal_line_length_limits, another_extremely_long_variable_name_for_demonstration_purposes
```

### SQL with Long Queries
```sql
-- This is a very long SQL comment that explains the query in great detail and might cause overflow issues if not handled properly
SELECT 
    very_long_column_name_that_exceeds_normal_limits,
    another_extremely_long_column_name_for_testing,
    yet_another_long_column_name_that_should_wrap_properly,
    CASE 
        WHEN very_long_condition_that_might_cause_overflow_issues_in_pdf_generation = 'very_long_value_that_should_wrap_properly' 
        THEN 'very_long_result_value_that_demonstrates_text_wrapping_capabilities'
        ELSE 'another_long_default_value_for_comprehensive_testing_purposes'
    END as very_long_alias_name_that_tests_wrapping
FROM very_long_table_name_that_might_cause_overflow_issues_in_code_blocks
WHERE very_long_column_name_that_exceeds_normal_limits LIKE '%very_long_search_pattern_that_should_wrap_properly%'
    AND another_extremely_long_column_name_for_testing IN ('value1_that_is_very_long', 'value2_that_is_also_very_long', 'value3_for_comprehensive_testing')
ORDER BY very_long_column_name_that_exceeds_normal_limits DESC, another_extremely_long_column_name_for_testing ASC;
```

## 📋 Long List Items Test

### Unordered Lists with Long Content
- This is a very long list item that contains extensive text content which should be properly wrapped and formatted without causing any overflow issues in the PDF generation process. It includes various types of content and should demonstrate proper text wrapping capabilities.
- Another long item with a URL: https://www.example.com/very/long/path/that/should/wrap/properly/in/list/items/without/causing/overflow/issues
- VeryLongWordWithoutSpacesThatShouldBreakProperlyInListItemsWithoutCausingOverflowIssues
- List item with `very_long_inline_code_that_should_wrap_properly_without_causing_issues` and regular text
- **Bold text with very long content that should wrap properly** and *italic text with equally long content for comprehensive testing purposes*

### Ordered Lists with Long Content
1. First numbered item with very long content that spans multiple lines and should wrap properly without causing any overflow issues in the PDF generation process
2. Second item with technical terms like supercalifragilisticexpialidocious and pneumonoultramicroscopicsilicovolcanoconiosiswhichisaverylongword
3. Third item with a very long URL: https://api.example.com/v1/users/12345/documents/reports/financial/quarterly/2023/q4/detailed-analysis-with-charts-and-graphs-and-comprehensive-data
4. Fourth item with mixed content including **bold very long text that should wrap properly**, *italic long text*, and `long_code_snippets_that_should_also_wrap_nicely`

## 💬 Long Blockquotes Test

> This is a very long blockquote that contains extensive text content which should be properly wrapped and formatted without causing any overflow issues in the PDF generation process. It includes various types of content like URLs (https://www.example.com/very/long/path/for/testing/purposes), long words (supercalifragilisticexpialidocious), and technical terms that should all be handled gracefully by the text wrapping system.

> Another blockquote with even more comprehensive content including code snippets like `very_long_function_name_that_might_cause_overflow_issues()`, mathematical symbols like ∑∏√∂∫, and special characters like © ® ™ that should all be properly formatted and wrapped without causing any layout issues in the PDF generation process.

## 📄 Long Paragraph Test

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem. This paragraph also includes a very long URL: https://www.example.com/very/long/path/that/should/wrap/properly/within/paragraph/text/without/causing/overflow/issues/in/pdf/generation/process and continues with more text to test comprehensive wrapping capabilities.

## ✅ Test Results Summary

If this PDF renders correctly with all the long content properly wrapped and no overflow issues, then the text overflow handling improvements are working as expected. The key features being tested include:

- **Long URL wrapping** in paragraphs, lists, tables, and code blocks
- **Very long word breaking** for technical terms and compound words
- **Table cell content wrapping** with fixed layout and proper text breaking
- **Code block wrapping** while preserving formatting
- **List item wrapping** for both ordered and unordered lists
- **Blockquote wrapping** with proper indentation
- **Mixed content handling** with various formatting elements

---

5. **Testing Results**
   
   > All tests passed successfully. The application is ready for deployment.

---

*This concludes the comprehensive markdown style test document.*

**Note:** This document should render with proper formatting including headers, lists, tables, code blocks, quotes, and various text styles.
        ''',
        'question.txt': 'Please provide an analysis of the construction adhesive market in the US'
    }

    save_dict_to_pdf_file(input_dict, 'test_comprehensive.pdf')
