# Content Submission Feature

## Overview
Added a user-friendly content submission feature that allows website visitors to contribute counter-narrative content to the re.inform platform.

## Implementation Details

### Location
The submission feature is implemented on the **Contribute page** (`contribute.html`) with additional call-to-action links on the homepage.

### Components Added

#### 1. Submit Counter-Narrative Card (Contribute Page)
- New highlighted card in the contribution options grid
- Positioned after the "Content Creation" card
- Contains a "Submit Content" button that opens the submission modal
- Styled with a distinctive green gradient background and border

#### 2. Modal Window
A full-screen modal overlay with a form containing:
- **Narrative to Counter** (required) - textarea for describing the extremist narrative
- **Summary of Counter-Argument** (required) - textarea for the counter-narrative strategy
- **Source Link**(s)* - URL input for the source of the narrative

#### 3. Homepage Integration
- Added "Submit Counter-Narrative" button to the Call-to-Action section
- Links directly to the Contribute page
- Responsive button styling for mobile devices

### Features

#### User Experience
- Click anywhere outside the modal to close it
- Form validation for required fields
- Success message upon submission
- Form reset after submission or cancellation
- Body scroll lock when modal is open

#### Responsive Design
- Mobile-optimized modal layout
- Flexible button sizing on smaller screens
- Touch-friendly form inputs

### Technical Implementation

#### JavaScript Functions
```javascript
openSubmissionModal()  - Opens the modal and locks scroll
closeSubmissionModal() - Closes modal, unlocks scroll, resets form
```

#### Form Handling
Currently logs submission data to console with TODO for backend integration:
```javascript
{
  narrative: "...",
  summary: "...",
  source: "..."
}
```

### Styling
- Modal uses semi-transparent overlay (rgba(0, 0, 0, 0.5))
- Highlighted card uses primary color border (#29683e)
- Gradient background for visual emphasis
- Consistent with existing design system

### Next Steps
1. **Backend Integration**: Connect form submission to a database or API endpoint
2. **Email Notifications**: Send submissions to moderators
3. **User Authentication**: Optional login for tracking contributions
4. **Content Moderation**: Admin interface for reviewing submissions
5. **Success Page**: Dedicated thank-you page after submission

## Files Modified

1. **website/contribute.html**
   - Added submission card
   - Added modal HTML structure
   - Added JavaScript for modal functionality
   - Added CSS for modal styling

2. **website/index.html**
   - Updated CTA section with submission button

3. **website/css/styles.css**
   - Added responsive styling for CTA buttons on mobile

## Testing Checklist
- [ ] Modal opens and closes correctly
- [ ] Form validation works for required fields
- [ ] URL validation for source link
- [ ] Click outside modal closes it
- [ ] Mobile responsiveness
- [ ] Form resets after submission
- [ ] Success message displays
- [ ] Console logs submission data

## Browser Compatibility
- Modern browsers (Chrome, Firefox, Safari, Edge)
- Mobile browsers (iOS Safari, Chrome Mobile)
- Requires JavaScript enabled
