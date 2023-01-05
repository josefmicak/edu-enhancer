using Common;

namespace DomainModel
{
    public class TestTemplate
    {
        public string TestNameIdentifier { get; set; } = default!;
        public string TestNumberIdentifier { get; set; } = default!;
        public string Title { get; set; } = default!;
        public EnumTypes.NegativePoints NegativePoints { get; set; } = EnumTypes.NegativePoints.Disabled;
        public double? MinimumPoints { get; set; }
        public Subject Subject { get; set; } = default!;
        public string OwnerLogin { get; set; } = default!;
        public User Owner { get; set; } = default!;
        public ICollection<QuestionTemplate> QuestionTemplateList { get; set; } = default!;
        public bool IsTestingData { get; set; } = false;
    }
}