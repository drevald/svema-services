-- Migration: Add bot user for AI-generated comments
-- This bot user will be used by the shot-captioning service to add AI-generated captions

-- Check if bot user already exists, if not create it
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM users WHERE username = 'bot-blip-image-captioning-base') THEN
        INSERT INTO users (username, password_hash, email)
        VALUES ('bot-blip-image-captioning-base', '', 'bot-blip-image-captioning-base@svema.ai');

        RAISE NOTICE 'Created bot user: bot-blip-image-captioning-base';
    ELSE
        RAISE NOTICE 'Bot user already exists: bot-blip-image-captioning-base';
    END IF;
END $$;
