import { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { Loader2, RotateCcw } from 'lucide-react'

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription
} from '@/components/ui/Dialog'
import Button from '@/components/ui/Button'
import { getVLMPrompts, updateVLMPrompts, VLMPromptsResponse } from '@/api/lightrag'
import { errorMessage } from '@/lib/utils'

interface VLMPromptDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export default function VLMPromptDialog({
  open,
  onOpenChange
}: VLMPromptDialogProps) {
  const { t } = useTranslation()
  const [isLoading, setIsLoading] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const [systemPrompt, setSystemPrompt] = useState('')
  const [userPrompt, setUserPrompt] = useState('')
  const [language, setLanguage] = useState('')
  const [originalData, setOriginalData] = useState<VLMPromptsResponse | null>(null)

  // Fetch prompts when dialog opens
  useEffect(() => {
    if (!open) return

    const fetchPrompts = async () => {
      setIsLoading(true)
      try {
        const data = await getVLMPrompts()
        setSystemPrompt(data.system_prompt)
        setUserPrompt(data.user_prompt)
        setLanguage(data.language)
        setOriginalData(data)
      } catch (err) {
        toast.error(t('documentPanel.vlmPrompt.errors.fetchFailed', { error: errorMessage(err) }))
      } finally {
        setIsLoading(false)
      }
    }

    fetchPrompts()
  }, [open, t])

  // Handle save
  const handleSave = async () => {
    setIsSaving(true)
    try {
      const data = await updateVLMPrompts({
        system_prompt: systemPrompt,
        user_prompt: userPrompt
      })
      setOriginalData(data)
      toast.success(t('documentPanel.vlmPrompt.saveSuccess'))
    } catch (err) {
      toast.error(t('documentPanel.vlmPrompt.errors.saveFailed', { error: errorMessage(err) }))
    } finally {
      setIsSaving(false)
    }
  }

  // Handle reset to defaults (send empty strings)
  const handleResetToDefaults = async () => {
    setIsSaving(true)
    try {
      const data = await updateVLMPrompts({
        system_prompt: '',
        user_prompt: ''
      })
      setSystemPrompt(data.system_prompt)
      setUserPrompt(data.user_prompt)
      setOriginalData(data)
      toast.success(t('documentPanel.vlmPrompt.resetSuccess'))
    } catch (err) {
      toast.error(t('documentPanel.vlmPrompt.errors.resetFailed', { error: errorMessage(err) }))
    } finally {
      setIsSaving(false)
    }
  }

  // Check if there are changes
  const hasChanges = originalData && (
    systemPrompt !== originalData.system_prompt ||
    userPrompt !== originalData.user_prompt
  )

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[800px] max-h-[85vh] overflow-hidden flex flex-col">
        <DialogDescription className="sr-only">
          {t('documentPanel.vlmPrompt.description')}
        </DialogDescription>
        <DialogHeader className="flex-shrink-0">
          <DialogTitle className="flex items-center gap-2">
            {t('documentPanel.vlmPrompt.title')}
            {language && (
              <span className="text-sm font-normal text-muted-foreground">
                ({t('documentPanel.vlmPrompt.language')}: {language})
              </span>
            )}
          </DialogTitle>
        </DialogHeader>

        {/* Content Area */}
        <div className="flex-1 overflow-y-auto space-y-4 mt-4" style={{ maxHeight: 'calc(85vh - 160px)' }}>
          {isLoading ? (
            <div className="flex items-center justify-center min-h-[200px]">
              <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
            </div>
          ) : (
            <>
              {/* System Prompt */}
              <div className="space-y-2">
                <label className="text-sm font-medium">
                  {t('documentPanel.vlmPrompt.systemPromptLabel')}
                </label>
                <textarea
                  value={systemPrompt}
                  onChange={(e) => setSystemPrompt(e.target.value)}
                  className="w-full h-40 p-3 text-sm font-mono rounded-md border bg-background resize-y"
                  placeholder={t('documentPanel.vlmPrompt.systemPromptPlaceholder')}
                />
                <p className="text-xs text-muted-foreground">
                  {t('documentPanel.vlmPrompt.systemPromptHelp')}
                </p>
              </div>

              {/* User Prompt */}
              <div className="space-y-2">
                <label className="text-sm font-medium">
                  {t('documentPanel.vlmPrompt.userPromptLabel')}
                </label>
                <textarea
                  value={userPrompt}
                  onChange={(e) => setUserPrompt(e.target.value)}
                  className="w-full h-24 p-3 text-sm font-mono rounded-md border bg-background resize-y"
                  placeholder={t('documentPanel.vlmPrompt.userPromptPlaceholder')}
                />
                <p className="text-xs text-muted-foreground">
                  {t('documentPanel.vlmPrompt.userPromptHelp')}
                </p>
              </div>
            </>
          )}
        </div>

        {/* Footer Actions */}
        <div className="flex justify-between items-center pt-4 border-t flex-shrink-0">
          <Button
            variant="outline"
            size="sm"
            onClick={handleResetToDefaults}
            disabled={isLoading || isSaving}
          >
            <RotateCcw className="h-4 w-4 mr-1" />
            {t('documentPanel.vlmPrompt.resetToDefaults')}
          </Button>
          <div className="flex gap-2">
            <Button
              variant="outline"
              onClick={() => onOpenChange(false)}
              disabled={isSaving}
            >
              {t('common.cancel')}
            </Button>
            <Button
              onClick={handleSave}
              disabled={isLoading || isSaving || !hasChanges}
            >
              {isSaving ? (
                <>
                  <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                  {t('common.saving')}
                </>
              ) : (
                t('common.save')
              )}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
