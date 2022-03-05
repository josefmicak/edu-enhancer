namespace ViewLayer.Controllers
{
    public interface IFilePathManager
    {
        public string GetFilePath(string testNameIdentifier, string testNumberIdentifier);
    }
}
