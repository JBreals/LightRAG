import { useState, useEffect, useMemo } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { Copy, Check, Loader2, FileText, Image, Table2, FunctionSquare, AlertCircle, Layers, ZoomIn, X } from 'lucide-react'

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogOverlay,
  DialogPortal
} from '@/components/ui/Dialog'
import * as DialogPrimitive from '@radix-ui/react-dialog'
import Button from '@/components/ui/Button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/Tabs'
import {
  getDocumentContent,
  getBlockMappings,
  getBlockMappingsByFilePath,
  getBlockImageThumbnailUrl,
  getBlockImageUrl,
  DocContentResponse,
  DocumentBlockMappings,
  BlockMapping
} from '@/api/lightrag'
import { errorMessage } from '@/lib/utils'
import { cn } from '@/lib/utils'

interface DocumentContentDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  docId: string | null
  fileName?: string
}

type BlockType = 'all' | 'text' | 'image' | 'table' | 'equation'

// Block type configuration
const blockTypeConfig: Record<BlockType, { icon: React.ElementType; label: string }> = {
  all: { icon: Layers, label: 'blockMapping.tabs.all' },
  text: { icon: FileText, label: 'blockMapping.tabs.text' },
  image: { icon: Image, label: 'blockMapping.tabs.image' },
  table: { icon: Table2, label: 'blockMapping.tabs.table' },
  equation: { icon: FunctionSquare, label: 'blockMapping.tabs.equation' }
}

// Single block item component for text/table/equation
function TextBlockItem({
  block
}: {
  block: BlockMapping
}) {
  const { t } = useTranslation()
  const isSuccess = block.status === 'success'
  const isFailed = block.status === 'failed'

  return (
    <div
      className={cn(
        'rounded-lg border p-3',
        isFailed && 'border-destructive/50 bg-destructive/5',
        !isSuccess && !isFailed && 'border-muted bg-muted/30'
      )}
    >
      {/* Block header */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono bg-muted px-2 py-0.5 rounded">
            #{block.block_index}
          </span>
          <span className="text-xs text-muted-foreground">
            {t('blockMapping.page')} {block.page_idx + 1}
          </span>
          <span
            className={cn(
              'text-xs px-2 py-0.5 rounded',
              block.block_type === 'text' && 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300',
              block.block_type === 'table' && 'bg-purple-100 text-purple-700 dark:bg-purple-900 dark:text-purple-300',
              block.block_type === 'equation' && 'bg-orange-100 text-orange-700 dark:bg-orange-900 dark:text-orange-300'
            )}
          >
            {block.block_type}
          </span>
          {isFailed && (
            <span className="text-xs text-destructive flex items-center gap-1">
              <AlertCircle className="h-3 w-3" />
              {t('blockMapping.failed')}
            </span>
          )}
        </div>
        {block.processing_time && (
          <span className="text-xs text-muted-foreground">
            {block.processing_time.toFixed(1)}s
          </span>
        )}
      </div>

      {/* Block content */}
      <div className="space-y-2">
        <div>
          <div className="text-xs text-muted-foreground mb-1">{t('blockMapping.original')}:</div>
          <pre className="text-xs bg-muted/50 rounded p-2 whitespace-pre-wrap break-words max-h-[150px] overflow-y-auto font-mono">
            {block.original_content || '-'}
          </pre>
        </div>
        <div>
          <div className="text-xs text-muted-foreground mb-1">{t('blockMapping.converted')}:</div>
          <div className="text-xs bg-muted/50 rounded p-2 max-h-[100px] overflow-y-auto">
            {block.converted_content || '-'}
          </div>
        </div>

        {/* Error message */}
        {block.error_message && (
          <div className="text-xs text-destructive bg-destructive/10 rounded p-2">
            {block.error_message}
          </div>
        )}

        {/* BBox info */}
        <div className="text-xs text-muted-foreground">
          bbox: [{block.bbox.map((v) => Math.round(v)).join(', ')}]
        </div>
      </div>
    </div>
  )
}

// Image block item component - larger display
function ImageBlockItem({
  block,
  docId,
  onImageClick
}: {
  block: BlockMapping
  docId: string
  onImageClick: (url: string) => void
}) {
  const { t } = useTranslation()
  const isSuccess = block.status === 'success'
  const isFailed = block.status === 'failed'

  return (
    <div
      className={cn(
        'rounded-lg border p-4',
        isFailed && 'border-destructive/50 bg-destructive/5',
        !isSuccess && !isFailed && 'border-muted bg-muted/30'
      )}
    >
      {/* Block header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono bg-muted px-2 py-0.5 rounded">
            #{block.block_index}
          </span>
          <span className="text-xs text-muted-foreground">
            {t('blockMapping.page')} {block.page_idx + 1}
          </span>
          <span className="text-xs px-2 py-0.5 rounded bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300">
            image
          </span>
          {isFailed && (
            <span className="text-xs text-destructive flex items-center gap-1">
              <AlertCircle className="h-3 w-3" />
              {t('blockMapping.failed')}
            </span>
          )}
        </div>
        {block.processing_time && (
          <span className="text-xs text-muted-foreground">
            VLM: {block.processing_time.toFixed(1)}s
          </span>
        )}
      </div>

      {/* Image and VLM output side by side */}
      {isSuccess && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Image section */}
          <div className="space-y-2">
            <div className="text-xs text-muted-foreground font-medium">{t('blockMapping.original')}:</div>
            <div
              className="relative group cursor-pointer rounded-lg border bg-white dark:bg-gray-900 p-2 flex items-center justify-center min-h-[200px]"
              onClick={() => onImageClick(getBlockImageUrl(docId, block.block_index))}
            >
              <img
                src={getBlockImageThumbnailUrl(docId, block.block_index, 400)}
                alt={`Block ${block.block_index}`}
                className="max-w-full max-h-[300px] object-contain rounded"
                onError={(e) => {
                  ;(e.target as HTMLImageElement).src = ''
                  ;(e.target as HTMLImageElement).alt = 'Image not available'
                }}
              />
              <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors rounded-lg flex items-center justify-center">
                <ZoomIn className="h-8 w-8 text-white opacity-0 group-hover:opacity-70 transition-opacity" />
              </div>
            </div>
            <div className="text-xs text-muted-foreground">
              {t('blockMapping.clickToEnlarge')}
            </div>
          </div>

          {/* VLM output section */}
          <div className="space-y-2">
            <div className="text-xs text-muted-foreground font-medium">{t('blockMapping.vlmOutput')}:</div>
            <div className="bg-muted/50 rounded-lg p-3 min-h-[200px] max-h-[350px] overflow-y-auto">
              <div className="text-sm whitespace-pre-wrap">
                {block.converted_content || '-'}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Failed or skipped state */}
      {!isSuccess && (
        <div className="space-y-2">
          {block.error_message && (
            <div className="text-sm text-destructive bg-destructive/10 rounded p-3">
              {block.error_message}
            </div>
          )}
          <div className="text-sm text-muted-foreground">
            {block.original_content || 'No image path'}
          </div>
        </div>
      )}

      {/* BBox info */}
      <div className="text-xs text-muted-foreground mt-3 pt-2 border-t">
        bbox: [{block.bbox.map((v) => Math.round(v)).join(', ')}]
      </div>
    </div>
  )
}

// Generic block item for "all" tab - compact view
function CompactBlockItem({
  block,
  docId,
  onImageClick
}: {
  block: BlockMapping
  docId: string
  onImageClick: (url: string) => void
}) {
  const { t } = useTranslation()
  const isImage = block.block_type === 'image'
  const isSuccess = block.status === 'success'
  const isFailed = block.status === 'failed'

  return (
    <div
      className={cn(
        'rounded-lg border p-3',
        isFailed && 'border-destructive/50 bg-destructive/5',
        !isSuccess && !isFailed && 'border-muted bg-muted/30'
      )}
    >
      {/* Block header */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono bg-muted px-2 py-0.5 rounded">
            #{block.block_index}
          </span>
          <span className="text-xs text-muted-foreground">
            {t('blockMapping.page')} {block.page_idx + 1}
          </span>
          <span
            className={cn(
              'text-xs px-2 py-0.5 rounded',
              block.block_type === 'text' && 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300',
              block.block_type === 'image' && 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300',
              block.block_type === 'table' && 'bg-purple-100 text-purple-700 dark:bg-purple-900 dark:text-purple-300',
              block.block_type === 'equation' && 'bg-orange-100 text-orange-700 dark:bg-orange-900 dark:text-orange-300'
            )}
          >
            {block.block_type}
          </span>
          {isFailed && (
            <span className="text-xs text-destructive flex items-center gap-1">
              <AlertCircle className="h-3 w-3" />
              {t('blockMapping.failed')}
            </span>
          )}
        </div>
        {block.processing_time && (
          <span className="text-xs text-muted-foreground">
            {block.processing_time.toFixed(1)}s
          </span>
        )}
      </div>

      {/* Block content */}
      <div className="space-y-2">
        {/* Image display - thumbnail in compact view */}
        {isImage && isSuccess && (
          <div className="flex gap-4">
            <div
              className="flex-shrink-0 cursor-pointer hover:opacity-80 transition-opacity"
              onClick={() => onImageClick(getBlockImageUrl(docId, block.block_index))}
            >
              <img
                src={getBlockImageThumbnailUrl(docId, block.block_index, 120)}
                alt={`Block ${block.block_index}`}
                className="max-w-[120px] max-h-[120px] rounded border object-contain bg-white"
                onError={(e) => {
                  ;(e.target as HTMLImageElement).style.display = 'none'
                }}
              />
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-xs text-muted-foreground mb-1">{t('blockMapping.vlmOutput')}:</div>
              <div className="text-xs bg-muted/50 rounded p-2 max-h-[100px] overflow-y-auto">
                {block.converted_content}
              </div>
            </div>
          </div>
        )}

        {/* Text/Table/Equation display */}
        {!isImage && (
          <div className="grid grid-cols-2 gap-2">
            <div>
              <div className="text-xs text-muted-foreground mb-1">{t('blockMapping.original')}:</div>
              <pre className="text-xs bg-muted/50 rounded p-2 whitespace-pre-wrap break-words max-h-[80px] overflow-y-auto font-mono">
                {block.original_content?.substring(0, 200) || '-'}
                {block.original_content && block.original_content.length > 200 ? '...' : ''}
              </pre>
            </div>
            <div>
              <div className="text-xs text-muted-foreground mb-1">{t('blockMapping.converted')}:</div>
              <div className="text-xs bg-muted/50 rounded p-2 max-h-[80px] overflow-y-auto">
                {block.converted_content || '-'}
              </div>
            </div>
          </div>
        )}

        {/* Error message */}
        {block.error_message && (
          <div className="text-xs text-destructive bg-destructive/10 rounded p-2">
            {block.error_message}
          </div>
        )}
      </div>
    </div>
  )
}

// Image preview modal - uses separate Radix Dialog to properly handle stacking
function ImagePreviewModal({
  imageUrl,
  onClose
}: {
  imageUrl: string | null
  onClose: () => void
}) {
  return (
    <DialogPrimitive.Root open={!!imageUrl} onOpenChange={(open) => !open && onClose()}>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-[100] bg-black/90" />
        <DialogPrimitive.Content
          className="fixed inset-0 z-[100] flex items-center justify-center p-4 focus:outline-none"
          onClick={onClose}
        >
          <div className="relative max-w-[90vw] max-h-[90vh]" onClick={(e) => e.stopPropagation()}>
            <img
              src={imageUrl || ''}
              alt="Preview"
              className="max-w-full max-h-[85vh] object-contain rounded-lg"
            />
            <DialogPrimitive.Close
              className="absolute -top-3 -right-3 bg-white text-black rounded-full w-10 h-10 flex items-center justify-center hover:bg-gray-200 shadow-lg cursor-pointer"
            >
              <X className="h-5 w-5" />
            </DialogPrimitive.Close>
          </div>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  )
}

export default function DocumentContentDialog({
  open,
  onOpenChange,
  docId,
  fileName
}: DocumentContentDialogProps) {
  const { t } = useTranslation()
  const [content, setContent] = useState<DocContentResponse | null>(null)
  const [blockMappings, setBlockMappings] = useState<DocumentBlockMappings | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [isLoadingBlocks, setIsLoadingBlocks] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)
  const [activeTab, setActiveTab] = useState<string>('content')
  const [previewImage, setPreviewImage] = useState<string | null>(null)

  // Fetch document content and block mappings when dialog opens
  useEffect(() => {
    if (!open || !docId) {
      setContent(null)
      setBlockMappings(null)
      setError(null)
      setActiveTab('content')
      return
    }

    const fetchData = async () => {
      setIsLoading(true)
      setIsLoadingBlocks(true)
      setError(null)

      // Fetch content first
      let contentData: DocContentResponse | null = null
      try {
        contentData = await getDocumentContent(docId)
        setContent(contentData)
      } catch (err) {
        const errMsg = errorMessage(err)
        setError(errMsg)
        toast.error(t('documentPanel.documentContent.errors.fetchFailed', { error: errMsg }))
        setIsLoading(false)
        setIsLoadingBlocks(false)
        return
      }
      setIsLoading(false)

      // Try to fetch block mappings by file_path first, then by doc_id
      try {
        let mappingsData: DocumentBlockMappings | null = null

        // Try by file_path from content response
        const filePath = contentData?.file_path || fileName
        if (filePath) {
          try {
            mappingsData = await getBlockMappingsByFilePath(filePath)
          } catch {
            // File path lookup failed, try doc_id
          }
        }

        // Fall back to doc_id if file_path lookup failed
        if (!mappingsData) {
          try {
            mappingsData = await getBlockMappings(docId)
          } catch {
            // doc_id lookup also failed
          }
        }

        setBlockMappings(mappingsData)
      } catch (err) {
        // Block mappings might not exist for non-RAG-Anything processed docs
        console.log('Block mappings not available:', err)
        setBlockMappings(null)
      } finally {
        setIsLoadingBlocks(false)
      }
    }

    fetchData()
  }, [open, docId, fileName, t])

  // Filter blocks by type
  const filteredBlocks = useMemo(() => {
    if (!blockMappings?.blocks) return []
    if (activeTab === 'all') return blockMappings.blocks
    return blockMappings.blocks.filter((b) => b.block_type === activeTab)
  }, [blockMappings, activeTab])

  // Handle copy to clipboard
  const handleCopy = async () => {
    if (!content?.content) return

    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(content.content)
      } else {
        const textArea = document.createElement('textarea')
        textArea.value = content.content
        textArea.style.position = 'fixed'
        textArea.style.left = '-999999px'
        textArea.style.top = '-999999px'
        document.body.appendChild(textArea)
        textArea.focus()
        textArea.select()
        document.execCommand('copy')
        document.body.removeChild(textArea)
      }
      setCopied(true)
      toast.success(t('documentPanel.documentContent.copySuccess'))
      setTimeout(() => setCopied(false), 2000)
    } catch (err) {
      console.error('Copy failed:', err)
      toast.error(t('documentPanel.documentContent.copyFailed'))
    }
  }

  const displayTitle = fileName || content?.file_path || docId || ''
  const hasBlockMappings = blockMappings && blockMappings.blocks.length > 0

  // Render block list based on active tab
  const renderBlockList = () => {
    if (isLoadingBlocks) {
      return (
        <div className="flex items-center justify-center h-full min-h-[300px]">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      )
    }

    if (filteredBlocks.length === 0) {
      return (
        <div className="flex items-center justify-center h-full min-h-[300px] text-muted-foreground">
          {t('blockMapping.noBlocks')}
        </div>
      )
    }

    // Use block mapping's doc_id for image URLs (RAG-Anything uses different doc_id)
    const mappingDocId = blockMappings?.doc_id || docId!

    // Image tab - use larger ImageBlockItem
    if (activeTab === 'image') {
      return (
        <div className="space-y-4">
          {filteredBlocks.map((block) => (
            <ImageBlockItem
              key={block.block_index}
              block={block}
              docId={mappingDocId}
              onImageClick={setPreviewImage}
            />
          ))}
        </div>
      )
    }

    // All tab - use compact view
    if (activeTab === 'all') {
      return (
        <div className="space-y-3">
          {/* Summary stats */}
          <div className="flex flex-wrap gap-4 mb-4 p-3 bg-muted/30 rounded-lg text-sm">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-blue-500" />
              <span>{t('blockMapping.tabs.text')}: {blockMappings!.text_blocks}</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-green-500" />
              <span>{t('blockMapping.tabs.image')}: {blockMappings!.image_blocks}</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-purple-500" />
              <span>{t('blockMapping.tabs.table')}: {blockMappings!.table_blocks}</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-orange-500" />
              <span>{t('blockMapping.tabs.equation')}: {blockMappings!.equation_blocks}</span>
            </div>
            {blockMappings!.failed_blocks > 0 && (
              <div className="flex items-center gap-2 text-destructive">
                <AlertCircle className="h-3 w-3" />
                <span>{t('blockMapping.failed')}: {blockMappings!.failed_blocks}</span>
              </div>
            )}
          </div>

          {/* Block list */}
          {filteredBlocks.map((block) => (
            <CompactBlockItem
              key={block.block_index}
              block={block}
              docId={mappingDocId}
              onImageClick={setPreviewImage}
            />
          ))}
        </div>
      )
    }

    // Text/Table/Equation tabs - use TextBlockItem
    return (
      <div className="space-y-3">
        {filteredBlocks.map((block) => (
          <TextBlockItem key={block.block_index} block={block} />
        ))}
      </div>
    )
  }

  return (
    <>
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent className="sm:max-w-[1100px] max-h-[90vh] overflow-hidden flex flex-col">
          <DialogDescription className="sr-only">
            {t('documentPanel.documentContent.description')}
          </DialogDescription>
          <DialogHeader className="flex flex-row items-center justify-between flex-shrink-0">
            <div className="flex-1 min-w-0 pr-8">
              <DialogTitle className="truncate" title={displayTitle}>
                {displayTitle}
              </DialogTitle>
              {content && (
                <p className="text-sm text-muted-foreground mt-1">
                  {t('documentPanel.documentContent.contentLength', {
                    length: content.content.length.toLocaleString()
                  })}
                  {blockMappings && (
                    <span className="ml-2">
                      • {blockMappings.parser.toUpperCase()} • {blockMappings.total_blocks}{' '}
                      {t('blockMapping.blocks')}
                    </span>
                  )}
                </p>
              )}
            </div>
            {content && (
              <Button variant="outline" size="sm" onClick={handleCopy} className="flex-shrink-0">
                {copied ? (
                  <>
                    <Check className="h-4 w-4 mr-1" />
                    {t('documentPanel.documentContent.copied')}
                  </>
                ) : (
                  <>
                    <Copy className="h-4 w-4 mr-1" />
                    {t('documentPanel.documentContent.copy')}
                  </>
                )}
              </Button>
            )}
          </DialogHeader>

          {/* Main content with tabs */}
          <div className="mt-4 flex-1 flex flex-col min-h-0 overflow-hidden">
            {isLoading ? (
              <div className="flex items-center justify-center flex-1 min-h-[300px]">
                <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
              </div>
            ) : error ? (
              <div className="flex items-center justify-center flex-1 min-h-[300px] text-destructive">
                {error}
              </div>
            ) : content ? (
              <Tabs
                value={activeTab}
                onValueChange={setActiveTab}
                className="flex-1 flex flex-col min-h-0 overflow-hidden"
              >
                <TabsList className="flex-shrink-0 w-full justify-start overflow-x-auto">
                  <TabsTrigger value="content" className="flex items-center gap-1">
                    <FileText className="h-4 w-4" />
                    {t('blockMapping.tabs.content')}
                  </TabsTrigger>
                  {hasBlockMappings && (
                    <>
                      {(Object.keys(blockTypeConfig) as BlockType[]).map((type) => {
                        const config = blockTypeConfig[type]
                        const Icon = config.icon
                        const count =
                          type === 'all'
                            ? blockMappings.total_blocks
                            : blockMappings[`${type}_blocks` as keyof DocumentBlockMappings] || 0
                        return (
                          <TabsTrigger
                            key={type}
                            value={type}
                            className="flex items-center gap-1"
                            disabled={type !== 'all' && count === 0}
                          >
                            <Icon className="h-4 w-4" />
                            {t(config.label)}
                            <span className="text-xs text-muted-foreground ml-1">({count as number})</span>
                          </TabsTrigger>
                        )
                      })}
                    </>
                  )}
                </TabsList>

                {/* Content tab */}
                <TabsContent value="content" className="flex-1 mt-4 min-h-0 overflow-y-auto data-[state=inactive]:hidden">
                  <div className="rounded-md border bg-muted/50 p-4">
                    <pre className="whitespace-pre-wrap break-words text-sm font-mono">
                      {content.content || t('documentPanel.documentContent.emptyContent')}
                    </pre>
                  </div>
                </TabsContent>

                {/* Block type tabs - render content only for active tab */}
                {hasBlockMappings && (
                  <>
                    {(Object.keys(blockTypeConfig) as BlockType[]).map((type) => (
                      <TabsContent key={type} value={type} className="flex-1 mt-4 min-h-0 overflow-y-auto data-[state=inactive]:hidden">
                        {activeTab === type && renderBlockList()}
                      </TabsContent>
                    ))}
                  </>
                )}
              </Tabs>
            ) : null}
          </div>
        </DialogContent>
      </Dialog>

      {/* Image preview modal */}
      <ImagePreviewModal imageUrl={previewImage} onClose={() => setPreviewImage(null)} />
    </>
  )
}
