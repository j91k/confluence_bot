# Obtaining a Confluence API Token

This guide will walk you through the process of creating an API token for your Confluence instance, which is required to run the Confluence Knowledge Chatbot.

## Steps to Get a Confluence API Token

### 1. Log in to your Atlassian Account

Go to [https://id.atlassian.com/](https://id.atlassian.com/) and log in with your Atlassian account credentials.

### 2. Navigate to Security Settings

Once logged in:
1. Click on your profile picture in the top right corner
2. Select "Account settings" from the dropdown menu
3. Navigate to the "Security" tab in the left sidebar
4. Look for "API tokens" section

### 3. Create a New API Token

1. Click the "Create API token" button
2. Enter a meaningful label for your token (e.g., "Confluence Knowledge Chatbot")
3. Select an expiration date for your token
4. Click "Create"
5. A new token will be generated and displayed on your screen

### 4. Copy and Store Your Token Securely

- Copy the generated token to a secure location
- **IMPORTANT:** This token will only be displayed once. If you lose it, you'll need to create a new one
- Never share your API token or commit it to version control

### 5. Use the Token with Your Application

When you run the application, you'll be prompted to enter your Confluence credentials through the user interface:
- Confluence URL (e.g., `https://your-domain.atlassian.net/wiki`)
- Confluence username (your email address)
- Confluence API token

**Note:** You do not need to add your Confluence API token to the `.env` file. The application will securely collect this information through its configuration interface.

## Additional Information

### Token Permissions

- The API token inherits the permissions of your Atlassian account
- Make sure your account has appropriate read access to the Confluence spaces you want to index

### Troubleshooting

If you encounter authentication issues:

1. Verify your Confluence URL format (e.g., `https://your-domain.atlassian.net/wiki`)
2. Ensure your username is the email address associated with your Atlassian account
3. Check that your account has the necessary permissions for the spaces you're trying to access
4. Try creating a new API token if problems persist

## Next Steps

Once you have your API token, return to the main README.md file and continue with the setup process.
